#include "cudaview/vk_engine.hpp"

#include "vk_properties.hpp"
#include "validation.hpp"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#include <set> // std::set
#include <stdexcept> // std::throw

static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;

VkFormat getVulkanFormat(DataFormat format)
{
  switch (format)
  {
    case DataFormat::Float32: return VK_FORMAT_R32_SFLOAT;
    case DataFormat::Rgba32:  return VK_FORMAT_R8G8B8A8_SRGB;
    default:                  return VK_FORMAT_UNDEFINED;
  }
}

VulkanEngine::VulkanEngine(int3 extent, cudaStream_t cuda_stream):
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
  point2d_pipeline(VK_NULL_HANDLE),
  point3d_pipeline(VK_NULL_HANDLE),
  screen_pipeline(VK_NULL_HANDLE),
  mesh_pipeline(VK_NULL_HANDLE),
  command_pool(VK_NULL_HANDLE),
  //vk_presentation_semaphore(VK_NULL_HANDLE),
  //vk_timeline_semaphore(VK_NULL_HANDLE),
  inflight_fences(MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE),
  image_available(MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE),
  render_finished(MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE),
  vk_wait_semaphore(VK_NULL_HANDLE),
  vk_signal_semaphore(VK_NULL_HANDLE),

  current_frame(0),

  window(nullptr),
  imgui_pool(VK_NULL_HANDLE),
  descriptor_pool(VK_NULL_HANDLE),
  rendering_modes{ {"structured", false}, {"unstructured", false} },

  texture_sampler(VK_NULL_HANDLE),
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
  if (imgui_pool != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorPool(device, imgui_pool, nullptr);
  }
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

  camera.type = Camera::CameraType::LookAt;
  //camera.flipY = true;
  camera.setPosition(glm::vec3(0.0f, 0.0f, -3.75f));
  camera.setRotation(glm::vec3(15.0f, 0.0f, 0.0f));
  camera.setRotationSpeed(0.5f);
  camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
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

      drawFrame();

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

    drawFrame();

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
  MappedUnstructuredMemory mapped{};
  mapped.element_count = elem_count;
  mapped.element_size  = elem_size;
  mapped.data_type     = type;
  mapped.data_domain   = domain;
  mapped.cuda_ptr      = nullptr;
  mapped.cuda_extmem   = nullptr;
  mapped.vk_format     = VK_FORMAT_UNDEFINED; //getVulkanFormat(format);
  mapped.vk_buffer     = VK_NULL_HANDLE;
  mapped.vk_memory     = VK_NULL_HANDLE;

  VkBufferUsageFlagBits usage;
  if (type == UnstructuredDataType::Points)
  {
    usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  }
  else if (type == UnstructuredDataType::Edges)
  {
    usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  }

  // Init unstructured memory
  createExternalBuffer(mapped.element_size * mapped.element_count,
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
  updateDescriptorSets();
  toggleRenderingMode("structured");
  *ptr_devmem = mapped.cuda_ptr;
}

void VulkanEngine::initVulkan()
{
  createCoreObjects();
  pickPhysicalDevice();
  createLogicalDevice();
  createCommandPool();
  createDescriptorSetLayout();
  createTextureSampler();

  initSwapchain();

  initImgui(); // After command pool and render pass are created
  createSyncObjects();
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
  uint32_t graphics_idx, present_idx;
  props::findQueueFamilies(physical_device, surface, graphics_idx, present_idx);

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::set unique_queue_families{ graphics_idx, present_idx};
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
  device_features.fillModeNonSolid  = VK_TRUE; // Enable wireframe

  // Explicitly enable timeline semaphores, or validation layer will complain
  VkPhysicalDeviceVulkan12Features features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features.timelineSemaphore = true;

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.pNext = &features;
  create_info.queueCreateInfoCount = queue_create_infos.size();
  create_info.pQueueCreateInfos    = queue_create_infos.data();
  create_info.pEnabledFeatures     = &device_features;

  auto device_extensions = props::getRequiredDeviceExtensions();
  create_info.enabledExtensionCount   = device_extensions.size();
  create_info.ppEnabledExtensionNames = device_extensions.data();

  if (validation::enable_layers)
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
  vkGetDeviceQueue(device, graphics_idx, 0, &graphics_queue);
  vkGetDeviceQueue(device, present_idx, 0, &present_queue);

  // TODO: Get device UUID
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

void VulkanEngine::handleMouseMove(float x, float y)
{
  auto dx = mouse_pos.x - x;
  auto dy = mouse_pos.y - y;

  if (mouse_buttons.left)
  {
    camera.rotate(glm::vec3(dy * camera.rotation_speed, -dx * camera.rotation_speed, 0.f));
    view_updated = true;
  }
  if (mouse_buttons.right)
  {
    camera.translate(glm::vec3(0.f, 0.f, dy * .005f));
  }
  if (mouse_buttons.middle)
  {
    camera.translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.f));
  }
  mouse_pos = glm::vec2(x, y);
}

void VulkanEngine::cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
  auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
  app->handleMouseMove(static_cast<float>(xpos), static_cast<float>(ypos));
}

void VulkanEngine::handleMouseButton(int button, int action)
{
  switch (action)
  {
    case GLFW_PRESS:
      switch (button)
      {
        case GLFW_MOUSE_BUTTON_LEFT:
          mouse_buttons.left = true;
          break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
          mouse_buttons.middle = true;
          break;
        case GLFW_MOUSE_BUTTON_RIGHT:
          mouse_buttons.right = true;
          break;
        default:
          break;
      }
      break;
    case GLFW_RELEASE:
      switch (button)
      {
        case GLFW_MOUSE_BUTTON_LEFT:
          mouse_buttons.left = false;
          break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
          mouse_buttons.middle = false;
          break;
        case GLFW_MOUSE_BUTTON_RIGHT:
          mouse_buttons.right = false;
          break;
        default:
          break;
      }
      break;
    default:
      break;
  }
}

void VulkanEngine::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
  auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
  app->handleMouseButton(button, action);
}

void VulkanEngine::framebufferResizeCallback(GLFWwindow *window, int width, int height)
{
  auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
  app->should_resize = true;
}
