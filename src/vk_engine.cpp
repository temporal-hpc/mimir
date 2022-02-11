#include "cudaview/vk_engine.hpp"
#include "cudaview/io.hpp"
#include "internal/camera.hpp"
#include "internal/vk_initializers.hpp"

#include "cudaview/vk_device.hpp"

#include "internal/vk_properties.hpp"
#include "internal/validation.hpp"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#include <set> // std::set
#include <stdexcept> // std::throw

VulkanEngine::VulkanEngine(int3 extent, cudaStream_t cuda_stream):
  current_frame(0),
  should_resize(false),
  shader_path(io::getDefaultShaderPath()),
  camera(std::make_unique<Camera>()),

  instance(VK_NULL_HANDLE),
  debug_messenger(VK_NULL_HANDLE),
  surface(VK_NULL_HANDLE),
  physical_device(VK_NULL_HANDLE),
  device(VK_NULL_HANDLE),
  graphics_queue(VK_NULL_HANDLE),
  present_queue(VK_NULL_HANDLE),

  swapchain(VK_NULL_HANDLE),
  render_pass(VK_NULL_HANDLE),
  descriptor_pool(VK_NULL_HANDLE),
  texture_sampler(VK_NULL_HANDLE),

  descriptor_layout(VK_NULL_HANDLE),
  pipeline_layout(VK_NULL_HANDLE),
  point2d_pipeline(VK_NULL_HANDLE),
  point3d_pipeline(VK_NULL_HANDLE),
  mesh2d_pipeline(VK_NULL_HANDLE),
  mesh3d_pipeline(VK_NULL_HANDLE),
  screen_pipeline(VK_NULL_HANDLE),

  vk_wait_semaphore(VK_NULL_HANDLE),
  vk_signal_semaphore(VK_NULL_HANDLE),
  //vk_presentation_semaphore(VK_NULL_HANDLE),
  //vk_timeline_semaphore(VK_NULL_HANDLE),
  cuda_wait_semaphore(nullptr),
  cuda_signal_semaphore(nullptr),
  //cuda_timeline_semaphore(nullptr),

  data_extent(extent),
  stream(cuda_stream),
  window(nullptr)
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
    validation::checkCuda(cudaStreamSynchronize(stream));
  }
  /*if (cuda_timeline_semaphore != nullptr)
  {
    validation::checkCuda(cudaDestroyExternalSemaphore(cuda_timeline_semaphore));
  }*/
  if (cuda_wait_semaphore != nullptr)
  {
    validation::checkCuda(cudaDestroyExternalSemaphore(cuda_wait_semaphore));
  }
  if (cuda_signal_semaphore != nullptr)
  {
    validation::checkCuda(cudaDestroyExternalSemaphore(cuda_signal_semaphore));
  }

  for (auto& buffer : unstructured_buffers)
  {
    if (buffer.cuda_ptr != nullptr)
    {
      validation::checkCuda(cudaDestroyExternalMemory(buffer.cuda_extmem));
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
      validation::checkCuda(cudaDestroyExternalMemory(buffer.cuda_extmem));
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
  if (descriptor_layout != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
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
  for (size_t i = 0; i < frames.size(); ++i)
  {
    if (frames[i].present_semaphore != VK_NULL_HANDLE)
    {
      vkDestroySemaphore(device, frames[i].present_semaphore, nullptr);
    }
    if (frames[i].render_semaphore != VK_NULL_HANDLE)
    {
      vkDestroySemaphore(device, frames[i].render_semaphore, nullptr);
    }
    if (frames[i].render_fence != VK_NULL_HANDLE)
    {
      vkDestroyFence(device, frames[i].render_fence, nullptr);
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

  dev.reset();
  if (validation::enable_layers)
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
  glfwSetCursorPosCallback(window, cursorPositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);

  initVulkan();

  camera->type = Camera::CameraType::LookAt;
  //camera->flipY = true;
  camera->setPosition(glm::vec3(0.0f, 0.0f, -3.75f));
  camera->setRotation(glm::vec3(15.0f, 0.0f, 0.0f));
  camera->setRotationSpeed(0.5f);
  camera->setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
                      //45.f, aspect_ratio, .1f, 10.f);
}

void VulkanEngine::displayAsync()
{
  rendering_thread = std::thread([this]()
  {
    while(!glfwWindowShouldClose(window))
    {
      glfwPollEvents(); // TODO: Move to main thread

      drawGui();

      std::unique_lock<std::mutex> lock(mutex);
      cond.wait(lock, [&]{ return device_working == false; });

      renderFrame();

      lock.unlock();
    }
    vkDeviceWaitIdle(device);
  });
}

void VulkanEngine::display(std::function<void(void)> func, size_t iter_count)
{
  size_t iteration_idx = 0;
  device_working = true;
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();

    drawGui();

    renderFrame();

    cudaSemaphoreWait();
    if (iteration_idx < iter_count)
    {
      // Advance the simulation
      func();
      iteration_idx++;
    }
    cudaSemaphoreSignal();
  }
  device_working = false;
  vkDeviceWaitIdle(device);
}

void VulkanEngine::registerUnstructuredMemory(void **ptr_devmem,
  size_t elem_count, size_t elem_size, UnstructuredDataType type,
  DataDomain domain)
{
  auto mapped = newUnstructuredMemory(elem_count, elem_size, type, domain);

  VkBufferUsageFlagBits usage;
  if (mapped.data_type == UnstructuredDataType::Points)
  {
    usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  }
  else if (mapped.data_type == UnstructuredDataType::Edges)
  {
    usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  }
  // Init unstructured memory
  dev->createExternalBuffer(mapped.element_size * mapped.element_count,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    mapped.vk_buffer, mapped.vk_memory
  );
  importCudaExternalMemory(&mapped.cuda_ptr, mapped.cuda_extmem,
    mapped.vk_memory, mapped.element_size * mapped.element_count
  );

  unstructured_buffers.push_back(mapped);
  updateDescriptorSets();
  *ptr_devmem = mapped.cuda_ptr;
}

void VulkanEngine::registerStructuredMemory(void **ptr_devmem,
  size_t width, size_t height, size_t elem_size, DataFormat format)
{
  auto mapped = newStructuredMemory(width, height, elem_size, format);

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
  updateDescriptorSets();
  *ptr_devmem = mapped.cuda_ptr;
}

void VulkanEngine::initVulkan()
{
  createCoreObjects();
  pickPhysicalDevice();
  dev = std::make_unique<VulkanDevice>(physical_device);
  dev->initLogicalDevice(surface);
  device = dev->logical_device;
  createLogicalDevice();
  createDescriptorSetLayout();
  createTextureSampler();

  initSwapchain();

  initImgui(); // After command pool and render pass are created
  createSyncObjects();
}

void VulkanEngine::createCoreObjects()
{
  if (validation::enable_layers &&
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

  auto extensions = props::getRequiredExtensions(validation::enable_layers);

  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pApplicationInfo = &app_info;
  instance_info.enabledExtensionCount   = extensions.size();
  instance_info.ppEnabledExtensionNames = extensions.data();

  VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
  // Include validation layer names if they are enabled
  if (validation::enable_layers)
  {
    auto debug_create_info = validation::debugMessengerCreateInfo();
    instance_info.pNext = &debug_create_info;
    instance_info.enabledLayerCount   = validation::layers.size();
    instance_info.ppEnabledLayerNames = validation::layers.data();
  }
  else
  {
    instance_info.pNext = nullptr;
    instance_info.enabledLayerCount   = 0;
    instance_info.ppEnabledLayerNames = nullptr;
  }

  //utils::listAvailableExtensions();
  validation::checkVulkan(vkCreateInstance(&instance_info, nullptr, &instance));

  if (validation::enable_layers)
  {
    // Details about the debug messenger and its callback
    auto debug_info = validation::debugMessengerCreateInfo();
    validation::checkVulkan(validation::CreateDebugUtilsMessengerEXT(
      instance, &debug_info, nullptr, &debug_messenger)
    );
  }

  validation::checkVulkan(
    glfwCreateWindowSurface(instance, window, nullptr, &surface)
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
    if (props::isDeviceSuitable(dev, surface))
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
  // Must be called after logical device is created (obviously!)
  vkGetDeviceQueue(device, dev->queue_indices.graphics, 0, &graphics_queue);
  vkGetDeviceQueue(device, dev->queue_indices.present, 0, &present_queue);

  // TODO: Get device UUID
}

void VulkanEngine::initImgui()
{
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForVulkan(window, true);

  ImGui_ImplVulkan_InitInfo init_info{};
  init_info.Instance       = instance;
  init_info.PhysicalDevice = physical_device;
  init_info.Device         = device;
  init_info.Queue          = graphics_queue;
  init_info.DescriptorPool = descriptor_pool;
  init_info.MinImageCount  = 3; // TODO: Check if this is true
  init_info.ImageCount     = 3;
  init_info.MSAASamples    = VK_SAMPLE_COUNT_1_BIT;
  ImGui_ImplVulkan_Init(&init_info, render_pass);

  auto cmd = dev->beginSingleTimeCommands();
  ImGui_ImplVulkan_CreateFontsTexture(cmd);
  dev->endSingleTimeCommands(cmd, graphics_queue);
  ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void VulkanEngine::importCudaExternalMemory(void **cuda_ptr,
  cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size)
{
  cudaExternalMemoryHandleDesc extmem_desc{};
  extmem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  extmem_desc.size = size;
  extmem_desc.handle.fd = (int)(uintptr_t)getMemoryHandle(
    vk_mem, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
  );

  validation::checkCuda(cudaImportExternalMemory(&cuda_mem, &extmem_desc));

  cudaExternalMemoryBufferDesc buffer_desc{};
  buffer_desc.offset = 0;
  buffer_desc.size = size;
  buffer_desc.flags = 0;

  validation::checkCuda(cudaExternalMemoryGetMappedBuffer(
    cuda_ptr, cuda_mem, &buffer_desc)
  );
}

void *VulkanEngine::getMemoryHandle(VkDeviceMemory memory,
  VkExternalMemoryHandleTypeFlagBits handle_type)
{
  int fd = -1;

  VkMemoryGetFdInfoKHR fd_info{};
  fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  fd_info.pNext = nullptr;
  fd_info.memory = memory;
  fd_info.handleType = handle_type;

  auto fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(
    device, "vkGetMemoryFdKHR"
  );
  if (!fpGetMemoryFdKHR)
  {
    throw std::runtime_error("Failed to retrieve function!");
  }
  if (fpGetMemoryFdKHR(device, &fd_info, &fd) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }
  return (void*)(uintptr_t)fd;
}
