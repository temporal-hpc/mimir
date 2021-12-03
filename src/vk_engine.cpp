#include "cudaview/vk_engine.hpp"
#include "vk_initializers.hpp"
#include "validation.hpp"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

//#include <iostream>
#include <cstring>
#include <experimental/source_location> // std::experimental::source_location
#include <limits> // std::numeric_limits
#include <set> // std::set
#include <stdexcept> // std::throw

#include "cudaview/vk_types.hpp"

static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;

using source_location = std::experimental::source_location;

constexpr void checkCuda(cudaError_t code, bool panic = true,
  source_location src = source_location::current())
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA assertion: %s on function %s at %s(%d)\n",
      cudaGetErrorString(code), src.function_name(), src.file_name(), src.line()
    );
    if (panic)
    {
      throw std::runtime_error("CUDA failure!");
    }
  }
}

VkFormat getVulkanFormat(DataFormat format)
{
  switch (format)
  {
    case DataFormat::Float32: return VK_FORMAT_R32_SFLOAT;
    case DataFormat::Rgba32:  return VK_FORMAT_R8G8B8A8_SRGB;
    default:                  return VK_FORMAT_UNDEFINED;
  }
}

static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
{
  auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
  app->should_resize = true;
}

VulkanEngine::VulkanEngine(int2 extent, cudaStream_t cuda_stream):
  should_resize(false),
  instance(VK_NULL_HANDLE),
  debug_messenger(VK_NULL_HANDLE),
  surface(VK_NULL_HANDLE),
  physical_device(VK_NULL_HANDLE),
  device(VK_NULL_HANDLE),
  graphics_queue(VK_NULL_HANDLE),
  present_queue(VK_NULL_HANDLE),
  swapchain(VK_NULL_HANDLE),
  render_pass(VK_NULL_HANDLE),
  descriptor_layout(VK_NULL_HANDLE),
  pipeline_layout(VK_NULL_HANDLE),
  graphics_pipeline(VK_NULL_HANDLE),
  screen_pipeline(VK_NULL_HANDLE),
  command_pool(VK_NULL_HANDLE),
  //vk_presentation_semaphore(VK_NULL_HANDLE),
  //vk_timeline_semaphore(VK_NULL_HANDLE),
  inflight_fences(MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE),
  image_available(MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE),
  render_finished(MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE),
  vk_wait_semaphore(VK_NULL_HANDLE),
  vk_signal_semaphore(VK_NULL_HANDLE),
  vertex_buffer(VK_NULL_HANDLE),
  vertex_buffer_memory(VK_NULL_HANDLE),
  index_buffer(VK_NULL_HANDLE),
  index_buffer_memory(VK_NULL_HANDLE),
  current_frame(0),

  window(nullptr),
  imgui_pool(VK_NULL_HANDLE),
  descriptor_pool(VK_NULL_HANDLE),
  rendering_modes{ {"structured", false}, {"unstructured", false} },

  staging_buffer(VK_NULL_HANDLE),
  staging_memory(VK_NULL_HANDLE),
  texture_sampler(VK_NULL_HANDLE),
  iteration_count(0),
  iteration_idx(0),
  data_extent(extent),
  stream(cuda_stream),
  //cuda_timeline_semaphore(nullptr),
  cuda_wait_semaphore(nullptr),
  cuda_signal_semaphore(nullptr)
{}

VulkanEngine::~VulkanEngine()
{
  if (rendering_thread.joinable())
  {
    rendering_thread.join();
  }
  vkDeviceWaitIdle(device);

  if (stream != nullptr)
  {
    checkCuda(cudaStreamSynchronize(stream));
  }
  /*if (cuda_timeline_semaphore != nullptr)
  {
    checkCuda(cudaDestroyExternalSemaphore(cuda_timeline_semaphore));
  }*/
  if (cuda_wait_semaphore != nullptr)
  {
    checkCuda(cudaDestroyExternalSemaphore(cuda_wait_semaphore));
  }
  if (cuda_signal_semaphore != nullptr)
  {
    checkCuda(cudaDestroyExternalSemaphore(cuda_signal_semaphore));
  }

  for (auto& buffer : unstructured_buffers)
  {
    if (buffer.cuda_ptr != nullptr)
    {
      checkCuda(cudaDestroyExternalMemory(buffer.cuda_extmem));
    }
    if (buffer.vk_buffer != VK_NULL_HANDLE)
    {
      vkDestroyBuffer(device, buffer.vk_buffer, nullptr);
    }
    if (buffer.vk_memory != VK_NULL_HANDLE)
    {
      vkFreeMemory(device, buffer.vk_memory, nullptr);
    }
  }

  for (auto& buffer : structured_buffers)
  {
    if (buffer.cuda_ptr != nullptr)
    {
      checkCuda(cudaDestroyExternalMemory(buffer.cuda_extmem));
    }
    if (buffer.vk_buffer != VK_NULL_HANDLE)
    {
      vkDestroyBuffer(device, buffer.vk_buffer, nullptr);
    }
    if (buffer.vk_memory != VK_NULL_HANDLE)
    {
      vkFreeMemory(device, buffer.vk_memory, nullptr);
    }
    if (buffer.vk_view != VK_NULL_HANDLE)
    {
      vkDestroyImageView(device, buffer.vk_view, nullptr);
    }
    if (buffer.vk_image != VK_NULL_HANDLE)
    {
      vkDestroyImage(device, buffer.vk_image, nullptr);
    }
  }

  cleanupSwapchain();

  ImGui_ImplVulkan_Shutdown();
  if (imgui_pool != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorPool(device, imgui_pool, nullptr);
  }
  if (descriptor_layout != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
  }
  if (vertex_buffer != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(device, vertex_buffer, nullptr);
  }
  if (vertex_buffer_memory != VK_NULL_HANDLE)
  {
    vkFreeMemory(device, vertex_buffer_memory, nullptr);
  }
  if (index_buffer != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(device, index_buffer, nullptr);
  }
  if (index_buffer_memory != VK_NULL_HANDLE)
  {
    vkFreeMemory(device, index_buffer_memory, nullptr);
  }
  if (texture_sampler != VK_NULL_HANDLE)
  {
    vkDestroySampler(device, texture_sampler, nullptr);
  }

  /*if (vk_presentation_semaphore != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(device, vk_presentation_semaphore, nullptr);
  }
  if (vk_timeline_semaphore != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(device, vk_timeline_semaphore, nullptr);
  }*/
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
  {
    if (image_available[i] != VK_NULL_HANDLE)
    {
      vkDestroySemaphore(device, image_available[i], nullptr);
    }
    if (render_finished[i] != VK_NULL_HANDLE)
    {
      vkDestroySemaphore(device, render_finished[i], nullptr);
    }
    if (inflight_fences[i] != VK_NULL_HANDLE)
    {
      vkDestroyFence(device, inflight_fences[i], nullptr);
    }
  }
  if (vk_wait_semaphore != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(device, vk_wait_semaphore, nullptr);
  }
  if (vk_signal_semaphore != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(device, vk_signal_semaphore, nullptr);
  }

  if (command_pool != VK_NULL_HANDLE)
  {
    vkDestroyCommandPool(device, command_pool, nullptr);
  }
  if (device != VK_NULL_HANDLE)
  {
    vkDestroyDevice(device, nullptr);
  }
  if (validation::enable_validation_layers)
  {
    validation::DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
  }
  // Surface must be destroyed before instance
  if (surface != VK_NULL_HANDLE)
  {
    vkDestroySurfaceKHR(instance, surface, nullptr);
  }
  if (instance != VK_NULL_HANDLE)
  {
    vkDestroyInstance(instance, nullptr);
  }
  if (window != nullptr)
  {
    glfwDestroyWindow(window);
  }
  glfwTerminate();
}

void VulkanEngine::init(int width, int height)
{
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  window = glfwCreateWindow(width, height, "Vulkan test", nullptr, nullptr);
  glfwSetWindowUserPointer(window, this);
  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

  initVulkan();
}

void VulkanEngine::displayAsync()
{
  rendering_thread = std::thread(&VulkanEngine::mainLoopThreaded, this);
}

void VulkanEngine::mainLoopThreaded()
{
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents(); // TODO: Move to main thread

    drawGui();

    std::unique_lock<std::mutex> lock(mutex);
    cond.wait(lock, [&]{ return device_working == false; });

    drawFrame();

    lock.unlock();
  }
  vkDeviceWaitIdle(device);
}

void VulkanEngine::display()
{
  device_working = true;
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();

    drawGui();

    drawFrame();

    cudaSemaphoreWait();
    if (iteration_idx < iteration_count)
    {
      // Advance the simulation
      step_function();
      iteration_idx++;
    }
    cudaSemaphoreSignal();
  }
  device_working = false;
  vkDeviceWaitIdle(device);
}

void VulkanEngine::registerUnstructuredMemory(void **ptr_devmem,
  size_t elem_count, size_t elem_size)
{
  MappedUnstructuredMemory mapped{};
  mapped.element_count = elem_count;
  mapped.element_size  = elem_size;
  mapped.cuda_ptr      = nullptr;
  mapped.cuda_extmem   = nullptr;
  mapped.vk_format     = VK_FORMAT_UNDEFINED; //getVulkanFormat(format);
  mapped.vk_buffer     = VK_NULL_HANDLE;
  mapped.vk_memory     = VK_NULL_HANDLE;

  // Init unstructured memory
  createExternalBuffer(mapped.element_size * mapped.element_count,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    mapped.vk_buffer, mapped.vk_memory
  );
  importCudaExternalMemory(&mapped.cuda_ptr, mapped.cuda_extmem,
    mapped.vk_memory, mapped.element_size * mapped.element_count
  );

  unstructured_buffers.push_back(mapped);
  updateDescriptors();
  toggleRenderingMode("unstructured");
  *ptr_devmem = mapped.cuda_ptr;
}

void VulkanEngine::registerStructuredMemory(void **ptr_devmem,
  size_t width, size_t height, size_t elem_size, DataFormat format)
{
  MappedStructuredMemory mapped{};
  mapped.element_count = width * height;
  mapped.element_size  = elem_size;
  mapped.cuda_ptr      = nullptr;
  mapped.cuda_extmem   = nullptr;
  mapped.vk_format     = getVulkanFormat(format);
  mapped.vk_buffer     = VK_NULL_HANDLE;
  mapped.vk_memory     = VK_NULL_HANDLE;
  mapped.vk_image      = VK_NULL_HANDLE;
  mapped.vk_view       = VK_NULL_HANDLE;
  mapped.extent        = { width, height };

  // Init structured memory
  createExternalImage(width, height, mapped.vk_format,
    VK_IMAGE_TILING_LINEAR,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    mapped.vk_image, mapped.vk_memory
  );
  importCudaExternalMemory(&mapped.cuda_ptr, mapped.cuda_extmem,
    mapped.vk_memory, mapped.element_size * width * height
  );

  transitionImageLayout(mapped.vk_image, mapped.vk_format,
    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
  );
  mapped.vk_view = createImageView(mapped.vk_image, mapped.vk_format);

  structured_buffers.push_back(mapped);
  updateDescriptors();
  toggleRenderingMode("structured");
  *ptr_devmem = mapped.cuda_ptr;
}

void VulkanEngine::registerFunction(std::function<void(void)> func,
  size_t iter_count)
{
  step_function = func;
  iteration_count = iter_count;
}

void VulkanEngine::initVulkan()
{
  createInstance();
  setupDebugMessenger();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createCommandPool();
  createDescriptorSetLayout();

  createTextureSampler();

  initSwapchain();

  initImgui(); // After command pool and render pass are created
  createVertexBuffer();
  createIndexBuffer();
  createSyncObjects();
}

void VulkanEngine::importCudaExternalMemory(void **cuda_ptr,
  cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size)
{
  cudaExternalMemoryHandleDesc extmem_desc{};
  extmem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  extmem_desc.size = size;
  extmem_desc.handle.fd = (int)(uintptr_t)getMemHandle(
    vk_mem, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
  );

  checkCuda(cudaImportExternalMemory(&cuda_mem, &extmem_desc));

  cudaExternalMemoryBufferDesc buffer_desc{};
  buffer_desc.offset = 0;
  buffer_desc.size = size;
  buffer_desc.flags = 0;

  checkCuda(cudaExternalMemoryGetMappedBuffer(cuda_ptr, cuda_mem, &buffer_desc));
}

void VulkanEngine::importCudaExternalSemaphore(
  cudaExternalSemaphore_t& cuda_sem, VkSemaphore& vk_sem)
{
  cudaExternalSemaphoreHandleDesc sem_desc{};
  //sem_desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  sem_desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  sem_desc.handle.fd = (int)(uintptr_t)getSemaphoreHandle(
    vk_sem, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
  );
  sem_desc.flags = 0;
  checkCuda(cudaImportExternalSemaphore(&cuda_sem, &sem_desc));
}

void VulkanEngine::cudaSemaphoreWait()
{
  cudaExternalSemaphoreWaitParams wait_params{};
  wait_params.flags = 0;
  wait_params.params.fence.value = 0;
  // Wait for Vulkan to complete its work
  checkCuda(cudaWaitExternalSemaphoresAsync(//&cuda_timeline_semaphore
    &cuda_wait_semaphore, &wait_params, 1, stream)
  );
}

void VulkanEngine::cudaSemaphoreSignal()
{
  cudaExternalSemaphoreSignalParams signal_params{};
  signal_params.flags = 0;
  signal_params.params.fence.value = 0;

  // Signal Vulkan to continue with the updated buffers
  checkCuda(cudaSignalExternalSemaphoresAsync(//&cuda_timeline_semaphore
    &cuda_signal_semaphore, &signal_params, 1, stream)
  );
}

void VulkanEngine::createInstance()
{
  if (validation::enable_validation_layers &&
     !validation::checkValidationLayerSupport())
  {
    throw std::runtime_error("validation layers requested, but not supported");
  }

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName   = "Vulkan test";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName        = "No engine";
  app_info.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion         = VK_API_VERSION_1_2;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  auto extensions = getRequiredExtensions();
  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();

  VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
  // Include validation layer names if they are enabled
  if (validation::enable_validation_layers)
  {
    create_info.enabledLayerCount = static_cast<uint32_t>(validation::layers.size());
    create_info.ppEnabledLayerNames = validation::layers.data();

    validation::populateDebugMessengerCreateInfo(debug_create_info);
    create_info.pNext = &debug_create_info;
  }
  else
  {
    create_info.enabledLayerCount = 0;
    create_info.pNext = nullptr;
  }

  //utils::listAvailableExtensions();
  validation::checkVulkan(vkCreateInstance(&create_info, nullptr, &instance));
}

void VulkanEngine::setupDebugMessenger()
{
  if (!validation::enable_validation_layers) return;

  // Details about the debug messenger and its callback
  VkDebugUtilsMessengerCreateInfoEXT create_info{};
  validation::populateDebugMessengerCreateInfo(create_info);

  validation::checkVulkan(validation::CreateDebugUtilsMessengerEXT(
    instance, &create_info, nullptr, &debug_messenger)
  );
}

void VulkanEngine::pickPhysicalDevice()
{
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
  if (device_count == 0)
  {
    throw std::runtime_error("failed to find GPUs with Vulkan support");
  }
  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

  for (const auto& dev : devices)
  {
    if (isDeviceSuitable(dev))
    {
      physical_device = dev;
      break;
    }
  }
  if (physical_device == VK_NULL_HANDLE)
  {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
}

void VulkanEngine::createLogicalDevice()
{
  uint32_t graphics_queue_idx, present_queue_idx;
  findQueueFamilies(physical_device, graphics_queue_idx, present_queue_idx);

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::set unique_queue_families{ graphics_queue_idx, present_queue_idx};
  auto queue_priority = 1.f;

  for (auto queue_family : unique_queue_families)
  {
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_family;
    queue_create_info.queueCount       = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_create_info);
  }

  VkPhysicalDeviceFeatures device_features{};
  device_features.samplerAnisotropy = VK_TRUE;

  // Must explicitly enable timeline semaphores, or validation layer will complain
  VkPhysicalDeviceVulkan12Features features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features.timelineSemaphore = true;

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.pNext = &features;
  create_info.queueCreateInfoCount = queue_create_infos.size();
  create_info.pQueueCreateInfos    = queue_create_infos.data();
  create_info.pEnabledFeatures     = &device_features;

  auto device_extensions = getRequiredDeviceExtensions();
  create_info.enabledExtensionCount   = device_extensions.size();
  create_info.ppEnabledExtensionNames = device_extensions.data();

  if (validation::enable_validation_layers)
  {
    create_info.enabledLayerCount   = validation::layers.size();
    create_info.ppEnabledLayerNames = validation::layers.data();
  }
  else
  {
    create_info.enabledLayerCount = 0;
  }

  validation::checkVulkan(vkCreateDevice(
    physical_device, &create_info, nullptr, &device)
  );

  // Must be called after logical device is created (obviously!)
  vkGetDeviceQueue(device, graphics_queue_idx, 0, &graphics_queue);
  vkGetDeviceQueue(device, present_queue_idx, 0, &present_queue);

  // TODO: Get device UUID
}

void VulkanEngine::createSurface()
{
  validation::checkVulkan(
    glfwCreateWindowSurface(instance, window, nullptr, &surface)
  );
}

void VulkanEngine::initImgui()
{
  VkDescriptorPoolSize pool_sizes[] =
  {
    { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
    { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
    { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
    { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
    { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
    { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
    { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
  };

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets       = 1000;
  pool_info.poolSizeCount = std::size(pool_sizes);
  pool_info.pPoolSizes    = pool_sizes;

  validation::checkVulkan(vkCreateDescriptorPool(
    device, &pool_info, nullptr, &imgui_pool)
  );

  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForVulkan(window, true);

  ImGui_ImplVulkan_InitInfo init_info{};
  init_info.Instance       = instance;
  init_info.PhysicalDevice = physical_device;
  init_info.Device         = device;
  init_info.Queue          = graphics_queue;
  init_info.DescriptorPool = imgui_pool;
  init_info.MinImageCount  = 3; // TODO: Check if this is true
  init_info.ImageCount     = 3;
  init_info.MSAASamples    = VK_SAMPLE_COUNT_1_BIT;
  ImGui_ImplVulkan_Init(&init_info, render_pass);

  auto cmd = beginSingleTimeCommands();
  ImGui_ImplVulkan_CreateFontsTexture(cmd);
  endSingleTimeCommands(cmd);
  ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void VulkanEngine::createDescriptorSetLayout()
{
  auto ubo_layout = vkinit::descriptorLayoutBinding(0, // binding
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT
  );
  auto extent_layout = vkinit::descriptorLayoutBinding(1, // binding
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT
  );
  auto sampler_layout = vkinit::descriptorLayoutBinding(2, // binding
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT
  );

  std::array bindings{ubo_layout, extent_layout, sampler_layout};

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = bindings.size();
  layout_info.pBindings    = bindings.data();

  validation::checkVulkan(vkCreateDescriptorSetLayout(
    device, &layout_info, nullptr, &descriptor_layout)
  );
}

bool VulkanEngine::toggleRenderingMode(const std::string& key)
{
  // TODO: Use c++-20 map::contains(key)
  auto search = rendering_modes.find(key);
  if (search != rendering_modes.end())
  {
    rendering_modes[key] = !rendering_modes[key];
    return true;
  }
  return false;
}

void VulkanEngine::prepareWindow()
{
  std::unique_lock<std::mutex> ul(mutex);
  device_working = true;
  cudaSemaphoreWait();
  ul.unlock();
  cond.notify_one();
}

void VulkanEngine::updateWindow()
{
  std::unique_lock<std::mutex> ul(mutex);
  device_working = false;
  cudaSemaphoreSignal();
  ul.unlock();
  cond.notify_one();
}

void VulkanEngine::updateDescriptors()
{
  for (size_t i = 0; i < descriptor_sets.size(); ++i)
  {
    // Write MVP matrix, scene info and texture samplers
    std::vector<VkWriteDescriptorSet> desc_writes;
    desc_writes.reserve(2 + structured_buffers.size());

    VkDescriptorBufferInfo mvp_info{};
    mvp_info.buffer = uniform_buffers[i];
    mvp_info.offset = 0;
    mvp_info.range  = sizeof(ModelViewProjection);

    auto write_mvp = vkinit::writeDescriptorBuffer(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptor_sets[i], &mvp_info, 0
    );
    desc_writes.push_back(write_mvp);

    VkDescriptorBufferInfo extent_info{};
    extent_info.buffer = uniform_buffers[i];
    extent_info.offset = sizeof(ModelViewProjection);
    extent_info.range  = sizeof(SceneParams);

    auto write_scene = vkinit::writeDescriptorBuffer(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptor_sets[i], &extent_info, 1
    );
    desc_writes.push_back(write_scene);

    for (const auto& buffer : structured_buffers)
    {
      VkDescriptorImageInfo img_info{};
      img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      img_info.imageView   = buffer.vk_view;
      img_info.sampler     = texture_sampler;

      auto write_tex = vkinit::writeDescriptorImage(
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptor_sets[i], &img_info, 2
      );
      desc_writes.push_back(write_tex);
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(desc_writes.size()),
      desc_writes.data(), 0, nullptr
    );
  }
}
