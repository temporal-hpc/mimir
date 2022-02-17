#include "cudaview/vk_engine.hpp"
#include "cudaview/io.hpp"

#include "internal/camera.hpp"
#include "internal/color.hpp"
#include "internal/validation.hpp"
#include "cudaview/engine/vk_device.hpp"
#include "cudaview/engine/vk_framebuffer.hpp"
#include "cudaview/engine/vk_initializers.hpp"
#include "cudaview/engine/vk_pipeline.hpp"
#include "cudaview/engine/vk_properties.hpp"
#include "cudaview/engine/vk_swapchain.hpp"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#include <filesystem> // std::filesystem
#include <set> // std::set
#include <stdexcept> // std::throw

VulkanEngine::VulkanEngine(int3 extent, cudaStream_t cuda_stream):
  shader_path(io::getDefaultShaderPath()),
  camera(std::make_unique<Camera>()),
  data_extent(extent),
  stream(cuda_stream)
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

  cleanupSwapchain();

  ImGui_ImplVulkan_Shutdown();
  deletors.flush();
  fbs.clear();
  swap.reset();
  dev.reset();
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
  auto buffer = dev->createExternalBuffer(mapped.element_size * mapped.element_count,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
  );
  mapped.vk_buffer = buffer.buffer;
  mapped.vk_memory = buffer.memory;
  importCudaExternalMemory(&mapped.cuda_ptr, mapped.cuda_extmem,
    mapped.vk_memory, mapped.element_size * mapped.element_count
  );
  deletors.pushFunction([=]{
    validation::checkCuda(cudaDestroyExternalMemory(mapped.cuda_extmem));
    vkDestroyBuffer(device, mapped.vk_buffer, nullptr);
    vkFreeMemory(device, mapped.vk_memory, nullptr);
  });

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

  deletors.pushFunction([=]{
    validation::checkCuda(cudaDestroyExternalMemory(mapped.cuda_extmem));
    vkDestroyBuffer(device, mapped.vk_buffer, nullptr);
    vkFreeMemory(device, mapped.vk_memory, nullptr);
    vkDestroyImageView(device, mapped.vk_view, nullptr);
    vkDestroyImage(device, mapped.vk_image, nullptr);
  });

  structured_buffers.push_back(mapped);
  updateDescriptorSets();
  *ptr_devmem = mapped.cuda_ptr;
}

void VulkanEngine::initVulkan()
{
  createInstance();
  swap = std::make_unique<VulkanSwapchain>();
  swap->initSurface(instance, window);
  pickPhysicalDevice();
  dev = std::make_unique<VulkanDevice>(physical_device);
  dev->initLogicalDevice(swap->surface);
  device = dev->logical_device;
  createDescriptorSetLayout();
  createDescriptorPool();
  createTextureSampler();

  initSwapchain();

  initImgui(); // After command pool and render pass are created
  createSyncObjects();
}

void VulkanEngine::createInstance()
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

  uint32_t glfw_ext_count = 0;
  // List required GLFW extensions and additional required validation layers
  const char **glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
  std::vector<const char*> extensions(glfw_exts, glfw_exts + glfw_ext_count);
  if (validation::enable_layers)
  {
    // Enable debugging message extension
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

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

  /*uint32_t extension_count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> available_exts(extension_count);
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, available_exts.data());

  std::cout << "Available extensions:\n";
  for (const auto& extension : available_exts)
  {
    std::cout << '\t' << extension.extensionName << '\n';
  }*/
  validation::checkVulkan(vkCreateInstance(&instance_info, nullptr, &instance));

  if (validation::enable_layers)
  {
    // Details about the debug messenger and its callback
    auto debug_info = validation::debugMessengerCreateInfo();
    validation::checkVulkan(validation::CreateDebugUtilsMessengerEXT(
      instance, &debug_info, nullptr, &debug_messenger)
    );
    deletors.pushFunction([=]{
      validation::DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
    });
  }
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
    if (props::isDeviceSuitable(dev, swap->surface))
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

void VulkanEngine::createDescriptorPool()
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
  pool_info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets       = 1000;
  pool_info.poolSizeCount = std::size(pool_sizes);
  pool_info.pPoolSizes    = pool_sizes;

  validation::checkVulkan(
    vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool)
  );
  deletors.pushFunction([=]{
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  });
}

void VulkanEngine::initImgui()
{
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForVulkan(window, true);

  ImGui_ImplVulkan_InitInfo init_info{};
  init_info.Instance       = instance;
  init_info.PhysicalDevice = physical_device;
  init_info.Device         = device;
  init_info.Queue          = dev->queues.graphics;
  init_info.DescriptorPool = descriptor_pool;
  init_info.MinImageCount  = 3; // TODO: Check if this is true
  init_info.ImageCount     = 3;
  init_info.MSAASamples    = VK_SAMPLE_COUNT_1_BIT;
  ImGui_ImplVulkan_Init(&init_info, render_pass);

  auto cmd = dev->beginSingleTimeCommands();
  ImGui_ImplVulkan_CreateFontsTexture(cmd);
  dev->endSingleTimeCommands(cmd, dev->queues.graphics);
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

void VulkanEngine::getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
  std::vector<VkPipelineStageFlags>& wait_stages) const
{
  // Wait semaphore has not been initialized on the first frame
  if (current_frame != 0 && device_working == true)
  {
    // Vulkan waits until Cuda is done with the display buffer before rendering
    wait.push_back(vk_wait_semaphore);
    // Cuda will wait until all pipeline commands are complete
    wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  }
}

void VulkanEngine::getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const
{
  // Vulkan will signal to this semaphore once the device array is ready
  // for Cuda to process
  signal.push_back(vk_signal_semaphore);
}

void VulkanEngine::createSyncObjects()
{
  images_inflight.resize(swap->image_count, VK_NULL_HANDLE);

  auto semaphore_info = vkinit::semaphoreCreateInfo();
  auto fence_info = vkinit::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
  for (auto& frame : frames)
  {
    validation::checkVulkan(vkCreateSemaphore(
      device, &semaphore_info, nullptr, &frame.present_semaphore)
    );
    validation::checkVulkan(vkCreateSemaphore(
      device, &semaphore_info, nullptr, &frame.render_semaphore)
    );
    validation::checkVulkan(vkCreateFence(
      device, &fence_info, nullptr, &frame.render_fence)
    );
    deletors.pushFunction([=]{
      vkDestroySemaphore(device, frame.present_semaphore, nullptr);
      vkDestroySemaphore(device, frame.render_semaphore, nullptr);
      vkDestroyFence(device, frame.render_fence, nullptr);
    });
  }

  /*validation::checkVulkan(vkCreateSemaphore(
    device, &semaphore_info, nullptr, &vk_presentation_semaphore)
  );
  createExternalSemaphore(vk_timeline_semaphore);
  importCudaExternalSemaphore(cuda_timeline_semaphore, vk_timeline_semaphore);
  if (cuda_timeline_semaphore != nullptr)
  {
    validation::checkCuda(cudaDestroyExternalSemaphore(cuda_timeline_semaphore));
  }
  if (vk_presentation_semaphore != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(device, vk_presentation_semaphore, nullptr);
  }
  if (vk_timeline_semaphore != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(device, vk_timeline_semaphore, nullptr);
  }*/

  createExternalSemaphore(vk_wait_semaphore);
  // Vulkan signal will be CUDA wait
  importCudaExternalSemaphore(cuda_signal_semaphore, vk_wait_semaphore);
  deletors.pushFunction([=]{
    vkDestroySemaphore(device, vk_wait_semaphore, nullptr);
    validation::checkCuda(cudaDestroyExternalSemaphore(cuda_signal_semaphore));
  });

  createExternalSemaphore(vk_signal_semaphore);
  // CUDA signal will be vulkan wait
  importCudaExternalSemaphore(cuda_wait_semaphore, vk_signal_semaphore);
  deletors.pushFunction([=]{
    vkDestroySemaphore(device, vk_signal_semaphore, nullptr);
    validation::checkCuda(cudaDestroyExternalSemaphore(cuda_wait_semaphore));
  });
}

void VulkanEngine::createExternalSemaphore(VkSemaphore& semaphore)
{
  /*VkSemaphoreTypeCreateInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_info.pNext = nullptr;
  timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_info.initialValue = 0;*/

  VkExportSemaphoreCreateInfoKHR export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
  export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
  export_info.pNext = nullptr; // &timeline_info

  auto semaphore_info = vkinit::semaphoreCreateInfo();
  semaphore_info.pNext = &export_info;

  validation::checkVulkan(
    vkCreateSemaphore(device, &semaphore_info, nullptr, &semaphore)
  );
}

void *VulkanEngine::getSemaphoreHandle(VkSemaphore semaphore,
  VkExternalSemaphoreHandleTypeFlagBits handle_type)
{
  int fd;
  VkSemaphoreGetFdInfoKHR fd_info{};
  fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  fd_info.pNext = nullptr;
  fd_info.semaphore  = semaphore;
  fd_info.handleType = handle_type;

  PFN_vkGetSemaphoreFdKHR fpGetSemaphore;
  fpGetSemaphore = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
    device, "vkGetSemaphoreFdKHR"
  );
  if (!fpGetSemaphore)
  {
    throw std::runtime_error("Failed to retrieve semaphore function handle!");
  }
  validation::checkVulkan(fpGetSemaphore(device, &fd_info, &fd));

  return (void*)(uintptr_t)fd;
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
  validation::checkCuda(cudaImportExternalSemaphore(&cuda_sem, &sem_desc));
}

void VulkanEngine::cudaSemaphoreWait()
{
  cudaExternalSemaphoreWaitParams wait_params{};
  wait_params.flags = 0;
  wait_params.params.fence.value = 0;
  // Wait for Vulkan to complete its work
  validation::checkCuda(cudaWaitExternalSemaphoresAsync(//&cuda_timeline_semaphore
    &cuda_wait_semaphore, &wait_params, 1, stream)
  );
}

void VulkanEngine::cudaSemaphoreSignal()
{
  cudaExternalSemaphoreSignalParams signal_params{};
  signal_params.flags = 0;
  signal_params.params.fence.value = 0;

  // Signal Vulkan to continue with the updated buffers
  validation::checkCuda(cudaSignalExternalSemaphoresAsync(//&cuda_timeline_semaphore
    &cuda_signal_semaphore, &signal_params, 1, stream)
  );
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


void VulkanEngine::cleanupSwapchain()
{
  vkDestroyBuffer(device, ubo.buffer, nullptr);
  vkFreeMemory(device, ubo.memory, nullptr);
  vkDestroyPipeline(device, point2d_pipeline, nullptr);
  vkDestroyPipeline(device, point3d_pipeline, nullptr);
  vkDestroyPipeline(device, mesh2d_pipeline, nullptr);
  vkDestroyPipeline(device, mesh3d_pipeline, nullptr);
  vkDestroyPipeline(device, screen_pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  vkDestroyRenderPass(device, render_pass, nullptr);
  vkFreeCommandBuffers(
    device, dev->command_pool, command_buffers.size(), command_buffers.data()
  );
  swap->cleanup();
}

void VulkanEngine::initSwapchain()
{
  int w, h;
  glfwGetFramebufferSize(window, &w, &h);
  uint32_t width = w;
  uint32_t height = h;
  std::vector<uint32_t> queue_indices{dev->queue_indices.graphics, dev->queue_indices.present};
  swap->create(width, height, queue_indices, dev->physical_device, dev->logical_device);
  command_buffers = dev->createCommandBuffers(swap->image_count);
  render_pass = createRenderPass(device, swap->color_format);

  auto images = swap->createImages(device);

  fbs.resize(swap->image_count);
  for (size_t i = 0; i < swap->image_count; ++i)
  {
    // Create a basic image view to be used as color target
    fbs[i].addAttachment(device, images[i], swap->color_format);
    fbs[i].create(device, render_pass, swap->swapchain_extent);
  }

  createGraphicsPipelines();
  createUniformBuffers();
  createDescriptorSets();
  updateDescriptorSets();
}

void VulkanEngine::recreateSwapchain()
{
  vkDeviceWaitIdle(device);

  cleanupSwapchain();
  initSwapchain();
}

// Take buffer with shader bytecode and create a shader module from it
VkShaderModule VulkanEngine::createShaderModule(const std::vector<char>& code)
{
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size();
  create_info.pCode    = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule module;
  validation::checkVulkan(vkCreateShaderModule(device, &create_info, nullptr, &module));

  return module;
}

void VulkanEngine::createGraphicsPipelines()
{
  auto orig_path = std::filesystem::current_path();
  std::filesystem::current_path(shader_path);

  auto vert_code   = io::readFile("shaders/unstructured/particle_pos_2d.spv");
  auto vert_module = createShaderModule(vert_code);

  auto frag_code   = io::readFile("shaders/unstructured/particle_draw.spv");
  auto frag_module = createShaderModule(frag_code);

  auto vert_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_VERTEX_BIT, vert_module
  );
  auto frag_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_FRAGMENT_BIT, frag_module
  );

  std::vector<VkDescriptorSetLayout> layouts{descriptor_layout};
  auto pipeline_layout_info = vkinit::pipelineLayoutCreateInfo(layouts);

  validation::checkVulkan(vkCreatePipelineLayout(
    device, &pipeline_layout_info, nullptr, &pipeline_layout)
  );

  std::vector<VkVertexInputBindingDescription> bind_desc;
  std::vector<VkVertexInputAttributeDescription> attr_desc;
  getVertexDescriptions2d(bind_desc, attr_desc);

  PipelineBuilder builder;
  builder.shader_stages.push_back(vert_info);
  builder.shader_stages.push_back(frag_info);
  builder.vertex_input_info = vkinit::vertexInputStateCreateInfo(bind_desc, attr_desc);
  builder.input_assembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
  builder.viewport.x        = 0.f;
  builder.viewport.y        = 0.f;
  builder.viewport.width    = static_cast<float>(swap->swapchain_extent.width);
  builder.viewport.height   = static_cast<float>(swap->swapchain_extent.height);
  builder.viewport.minDepth = 0.f;
  builder.viewport.maxDepth = 1.f;
  builder.scissor.offset    = {0, 0};
  builder.scissor.extent    = swap->swapchain_extent;
  builder.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
  builder.multisampling     = vkinit::multisamplingStateCreateInfo();
  builder.color_blend_attachment = vkinit::colorBlendAttachmentState();
  builder.pipeline_layout   = pipeline_layout;
  point2d_pipeline = builder.buildPipeline(device, render_pass);

  vkDestroyShaderModule(device, vert_module, nullptr);

  vert_code   = io::readFile("shaders/unstructured/particle_pos_3d.spv");
  vert_module = createShaderModule(vert_code);

  vert_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_VERTEX_BIT, vert_module
  );

  getVertexDescriptions3d(bind_desc, attr_desc);
  builder.shader_stages.clear();
  builder.shader_stages.push_back(vert_info);
  builder.shader_stages.push_back(frag_info);
  builder.vertex_input_info = vkinit::vertexInputStateCreateInfo(bind_desc, attr_desc);
  point3d_pipeline = builder.buildPipeline(device, render_pass);

  vkDestroyShaderModule(device, vert_module, nullptr);
  vkDestroyShaderModule(device, frag_module, nullptr);

  vert_code   = io::readFile("shaders/structured/screen_triangle.spv");
  vert_module = createShaderModule(vert_code);

  frag_code   = io::readFile("shaders/structured/texture_greyscale.spv");
  frag_module = createShaderModule(frag_code);

  vert_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_VERTEX_BIT, vert_module
  );
  frag_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_FRAGMENT_BIT, frag_module
  );

  builder.shader_stages.clear();
  builder.shader_stages.push_back(vert_info);
  builder.shader_stages.push_back(frag_info);
  builder.vertex_input_info = vkinit::vertexInputStateCreateInfo();
  builder.input_assembly    = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  screen_pipeline = builder.buildPipeline(device, render_pass);

  vkDestroyShaderModule(device, vert_module, nullptr);
  vkDestroyShaderModule(device, frag_module, nullptr);

  vert_code   = io::readFile("shaders/unstructured/wireframe_vertex_2d.spv");
  vert_module = createShaderModule(vert_code);

  frag_code   = io::readFile("shaders/unstructured/wireframe_fragment.spv");
  frag_module = createShaderModule(frag_code);

  vert_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_VERTEX_BIT, vert_module
  );
  frag_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_FRAGMENT_BIT, frag_module
  );

  getVertexDescriptions2d(bind_desc, attr_desc);
  builder.shader_stages.clear();
  builder.shader_stages.push_back(vert_info);
  builder.shader_stages.push_back(frag_info);
  builder.vertex_input_info = vkinit::vertexInputStateCreateInfo(bind_desc, attr_desc);
  builder.input_assembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  builder.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_LINE);
  mesh2d_pipeline = builder.buildPipeline(device, render_pass);

  vkDestroyShaderModule(device, vert_module, nullptr);

  vert_code   = io::readFile("shaders/unstructured/wireframe_vertex_3d.spv");
  vert_module = createShaderModule(vert_code);
  vert_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_VERTEX_BIT, vert_module
  );

  getVertexDescriptions3d(bind_desc, attr_desc);
  builder.shader_stages.clear();
  builder.shader_stages.push_back(vert_info);
  builder.shader_stages.push_back(frag_info);
  mesh3d_pipeline = builder.buildPipeline(device, render_pass);

  vkDestroyShaderModule(device, vert_module, nullptr);
  vkDestroyShaderModule(device, frag_module, nullptr);

  // Restore original working directory
  std::filesystem::current_path(orig_path);
}

void VulkanEngine::getVertexDescriptions2d(
  std::vector<VkVertexInputBindingDescription>& bind_desc,
  std::vector<VkVertexInputAttributeDescription>& attr_desc)
{
  bind_desc.resize(1);
  bind_desc[0].binding = 0;
  bind_desc[0].stride = sizeof(float2);
  bind_desc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  attr_desc.resize(1);
  attr_desc[0].binding = 0;
  attr_desc[0].location = 0;
  attr_desc[0].format = VK_FORMAT_R32G32_SFLOAT;
  attr_desc[0].offset = 0;
}

void VulkanEngine::getVertexDescriptions3d(
  std::vector<VkVertexInputBindingDescription>& bind_desc,
  std::vector<VkVertexInputAttributeDescription>& attr_desc)
{
  bind_desc.resize(1);
  bind_desc[0].binding = 0;
  bind_desc[0].stride = sizeof(float3);
  bind_desc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  attr_desc.resize(1);
  attr_desc[0].binding = 0;
  attr_desc[0].location = 0;
  attr_desc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attr_desc[0].offset = 0;
}

size_t getAlignedUniformSize(size_t original_size, size_t min_alignment)
{
	// Calculate required alignment based on minimum device offset alignment
	size_t aligned_size = original_size;
	if (min_alignment > 0) {
		aligned_size = (aligned_size + min_alignment - 1) & ~(min_alignment - 1);
	}
	return aligned_size;
}

void VulkanEngine::createUniformBuffers()
{
  auto min_alignment = dev->properties.limits.minUniformBufferOffsetAlignment;
  auto size_mvp = getAlignedUniformSize(sizeof(ModelViewProjection), min_alignment);
  auto size_colors = getAlignedUniformSize(sizeof(ColorParams), min_alignment);
  auto size_scene = getAlignedUniformSize(sizeof(SceneParams), min_alignment);

  auto img_count = swap->image_count;
  VkDeviceSize buffer_size = img_count * (size_mvp + size_colors + size_scene);
  ubo = dev->createBuffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  );
}

void VulkanEngine::updateUniformBuffer(uint32_t image_idx)
{
  auto min_alignment = dev->properties.limits.minUniformBufferOffsetAlignment;
  auto size_mvp = getAlignedUniformSize(sizeof(ModelViewProjection), min_alignment);
  auto size_colors = getAlignedUniformSize(sizeof(ColorParams), min_alignment);
  auto size_scene = getAlignedUniformSize(sizeof(SceneParams), min_alignment);
  auto size_ubo = size_mvp + size_colors + size_scene;
  auto offset = image_idx * size_ubo;

  ModelViewProjection mvp{};
  mvp.model = glm::mat4(1.f);
  mvp.view  = camera->matrices.view; // glm::mat4(1.f);
  mvp.proj  = camera->matrices.perspective; //glm::mat4(1.f);

  ColorParams colors{};
  colors.point_color = color::getColor(point_color);
  colors.edge_color  = color::getColor(edge_color);

  SceneParams params{};
  params.extent = glm::ivec3{data_extent.x, data_extent.y, data_extent.z};

  char *data = nullptr;
  vkMapMemory(device, ubo.memory, offset, size_ubo, 0, (void**)&data);
  std::memcpy(data, &mvp, sizeof(mvp));
  std::memcpy(data + size_mvp, &colors, sizeof(colors));
  std::memcpy(data + size_mvp + size_colors, &params, sizeof(params));
  vkUnmapMemory(device, ubo.memory);
}

void VulkanEngine::createDescriptorSets()
{
  auto img_count = swap->image_count;
  descriptor_sets.resize(img_count);

  std::vector<VkDescriptorSetLayout> layouts(img_count, descriptor_layout);
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool     = descriptor_pool;
  alloc_info.descriptorSetCount = static_cast<uint32_t>(img_count);
  alloc_info.pSetLayouts        = layouts.data();

  validation::checkVulkan(
    vkAllocateDescriptorSets(device, &alloc_info, descriptor_sets.data())
  );
}

void VulkanEngine::createDescriptorSetLayout()
{
  auto ubo_layout = vkinit::descriptorLayoutBinding(0, // binding
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT
  );
  auto extent_layout = vkinit::descriptorLayoutBinding(1, // binding
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT
  );
  auto point_color_layout = vkinit::descriptorLayoutBinding(2, // binding
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT
  );
  auto sampler_layout = vkinit::descriptorLayoutBinding(3, // binding
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT
  );

  std::array bindings{ubo_layout, extent_layout, point_color_layout, sampler_layout};

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = bindings.size();
  layout_info.pBindings    = bindings.data();

  validation::checkVulkan(vkCreateDescriptorSetLayout(
    device, &layout_info, nullptr, &descriptor_layout)
  );
	deletors.pushFunction([=](){
		vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
	});
}

void VulkanEngine::updateDescriptorSets()
{
  auto min_alignment = dev->properties.limits.minUniformBufferOffsetAlignment;
  auto size_mvp = getAlignedUniformSize(sizeof(ModelViewProjection), min_alignment);
  auto size_colors = getAlignedUniformSize(sizeof(ColorParams), min_alignment);
  auto size_scene = getAlignedUniformSize(sizeof(SceneParams), min_alignment);
  auto size_ubo = size_mvp + size_colors + size_scene;

  for (size_t i = 0; i < descriptor_sets.size(); ++i)
  {
    // Write MVP matrix, scene info and texture samplers
    std::vector<VkWriteDescriptorSet> desc_writes;
    desc_writes.reserve(3 + structured_buffers.size());

    VkDescriptorBufferInfo mvp_info{};
    mvp_info.buffer = ubo.buffer;
    mvp_info.offset = i * size_ubo;
    mvp_info.range  = sizeof(ModelViewProjection);

    auto write_mvp = vkinit::writeDescriptorBuffer(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptor_sets[i], &mvp_info, 0
    );
    desc_writes.push_back(write_mvp);

    VkDescriptorBufferInfo pcolor_info{};
    pcolor_info.buffer = ubo.buffer;
    pcolor_info.offset = i * size_ubo + size_mvp;
    pcolor_info.range  = sizeof(ColorParams);

    auto write_pcolor = vkinit::writeDescriptorBuffer(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptor_sets[i], &pcolor_info, 2
    );
    desc_writes.push_back(write_pcolor);

    VkDescriptorBufferInfo extent_info{};
    extent_info.buffer = ubo.buffer;
    extent_info.offset = i * size_ubo + size_mvp + size_colors;
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
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptor_sets[i], &img_info, 3
      );
      desc_writes.push_back(write_tex);
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(desc_writes.size()),
      desc_writes.data(), 0, nullptr
    );
  }
}

void VulkanEngine::drawGui()
{
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  //ImGui::ShowDemoWindow();
  ImGui::Render();
}

FrameData& VulkanEngine::getCurrentFrame()
{
  return frames[current_frame % frames.size()];
}

void VulkanEngine::renderFrame()
{
  constexpr auto timeout = std::numeric_limits<uint64_t>::max();
  /*const uint64_t wait_value = 0;
  const uint64_t signal_value = 1;

  VkSemaphoreWaitInfo wait_info{};
  wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  wait_info.pSemaphores = &vk_timeline_semaphore;
  wait_info.semaphoreCount = 1;
  wait_info.pValues = &wait_value;
  vkWaitSemaphores(device, &wait_info, timeout);*/

  auto frame = getCurrentFrame();
  vkWaitForFences(device, 1, &frame.render_fence, VK_TRUE, timeout);

  // Acquire image from swap chain
  uint32_t image_idx;
  // TODO: vk_presentation_semaphore instead of frame.present_semaphore
  auto result = vkAcquireNextImageKHR(device, swap->swapchain, timeout,
    frame.present_semaphore, VK_NULL_HANDLE, &image_idx
  );
  if (result == VK_ERROR_OUT_OF_DATE_KHR)
  {
    recreateSwapchain();
  }
  else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
  {
    throw std::runtime_error("Failed to acquire swapchain image");
  }

  if (images_inflight[image_idx] != VK_NULL_HANDLE)
  {
    vkWaitForFences(device, 1, &images_inflight[image_idx], VK_TRUE, timeout);
  }
  images_inflight[image_idx] = frame.render_fence;

  auto cmd_flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  auto begin_info = vkinit::commandBufferBeginInfo(cmd_flags);

  auto cmd = command_buffers[image_idx];
  validation::checkVulkan(vkBeginCommandBuffer(cmd, &begin_info));

  auto render_pass_info = vkinit::renderPassBeginInfo(
    render_pass, swap->swapchain_extent, fbs[image_idx].framebuffer
  );
  VkClearValue clear_color;
  color::setColor(clear_color.color.float32, bg_color);
  render_pass_info.clearValueCount = 1;
  render_pass_info.pClearValues    = &clear_color;

  vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

  drawObjects(image_idx);
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

  vkCmdEndRenderPass(cmd);
  // Finalize command buffer recording, so it can be executed
  validation::checkVulkan(vkEndCommandBuffer(cmd));

  updateUniformBuffer(image_idx);

  std::vector<VkSemaphore> wait_semaphores;
  std::vector<VkPipelineStageFlags> wait_stages;
  wait_semaphores.push_back(frame.present_semaphore); //vk_timeline_semaphore
  wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
  getWaitFrameSemaphores(wait_semaphores, wait_stages);

  std::vector<VkSemaphore> signal_semaphores;
  getSignalFrameSemaphores(signal_semaphores);
  signal_semaphores.push_back(frame.render_semaphore);//vk_timeline_semaphore

  auto submit_info = vkinit::submitInfo(&cmd);
  submit_info.waitSemaphoreCount   = wait_semaphores.size();
  submit_info.pWaitSemaphores      = wait_semaphores.data();
  submit_info.pWaitDstStageMask    = wait_stages.data();
  submit_info.signalSemaphoreCount = signal_semaphores.size();
  submit_info.pSignalSemaphores    = signal_semaphores.data();

  /*VkTimelineSemaphoreSubmitInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timeline_info.waitSemaphoreValueCount = 1;
  timeline_info.pWaitSemaphoreValues = &wait_value;
  timeline_info.signalSemaphoreValueCount = 1;
  timeline_info.pSignalSemaphoreValues = &signal_value;
  submit_info.pNext = &timeline_info;*/

  vkResetFences(device, 1, &frame.render_fence);

  // Execute command buffer using image as attachment in framebuffer
  validation::checkVulkan(vkQueueSubmit(
    dev->queues.graphics, 1, &submit_info, frame.render_fence) //VK_NULL_HANDLE
  );

  // Return image result back to swapchain for presentation on screen
  auto present_info = vkinit::presentInfo();
  present_info.swapchainCount     = 1;
  present_info.pSwapchains        = &swap->swapchain;
  present_info.waitSemaphoreCount = 1;
  //present_info.pWaitSemaphores = &vk_presentation_semaphore;
  present_info.pWaitSemaphores    = &frame.render_semaphore;
  present_info.pImageIndices      = &image_idx;

  result = vkQueuePresentKHR(dev->queues.present, &present_info);
  // Resize should be done after presentation to ensure semaphore consistency
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR
    || should_resize)
  {
    recreateSwapchain();
    should_resize = false;
  }

  current_frame++;
}

void VulkanEngine::drawObjects(uint32_t image_idx)
{
  auto cmd = command_buffers[image_idx];
  for (const auto& buffer : structured_buffers)
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, screen_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
      pipeline_layout, 0, 1, &descriptor_sets[image_idx], 0, nullptr
    );
    vkCmdDraw(cmd, 3, 1, 0, 0); // Draw a screen-covering triangle
  }

  for (const auto& buffer : unstructured_buffers)
  {
    if (buffer.data_type == UnstructuredDataType::Points)
    {
      if (buffer.data_domain == DataDomain::Domain2D)
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, point2d_pipeline);
      }
      else if (buffer.data_domain == DataDomain::Domain3D)
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, point3d_pipeline);
      }
      // Note: Second parameter can be also used to bind a compute pipeline
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline_layout, 0, 1, &descriptor_sets[image_idx], 0, nullptr
      );
      VkBuffer vertex_buffers[] = { buffer.vk_buffer };
      VkDeviceSize offsets[] = { 0 };
      auto binding_count = sizeof(vertex_buffers) / sizeof(vertex_buffers[0]);
      vkCmdBindVertexBuffers(cmd, 0, binding_count, vertex_buffers, offsets);
      vkCmdDraw(cmd, buffer.element_count, 1, 0, 0);
    }
    else if (buffer.data_type == UnstructuredDataType::Edges)
    {
      if (buffer.data_domain == DataDomain::Domain2D)
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh2d_pipeline);
      }
      else if (buffer.data_domain == DataDomain::Domain3D)
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh3d_pipeline);
      }
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline_layout, 0, 1, &descriptor_sets[image_idx], 0, nullptr
      );
      VkBuffer vertexBuffers[] = { unstructured_buffers[0].vk_buffer };
      VkDeviceSize offsets[] = {0};
      vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
      vkCmdBindIndexBuffer(cmd, buffer.vk_buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(cmd, 3 * buffer.element_count, 1, 0, 0, 0);
    }
  }
}
