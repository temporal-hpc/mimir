#include "cudaview/vk_engine.hpp"
#include "validation.hpp"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#include <limits> // std::numeric_limits
#include <set> // std::set
#include <stdexcept> // std::throw

#include "cudaview/vk_types.hpp"

static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;

static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
{
  auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
  app->should_resize = true;
}

VulkanEngine::VulkanEngine():
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
  screen_layout(VK_NULL_HANDLE),
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
  texture_sampler(VK_NULL_HANDLE)
{}

VulkanEngine::~VulkanEngine()
{
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
  uint32_t graphics_queue_index, present_queue_index;
  findQueueFamilies(physical_device, graphics_queue_index, present_queue_index);

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::set<uint32_t> unique_queue_families =
    { graphics_queue_index, present_queue_index};
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
  create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
  create_info.pQueueCreateInfos    = queue_create_infos.data();
  create_info.pEnabledFeatures     = &device_features;
  create_info.pNext = &features;

  auto device_extensions = getRequiredDeviceExtensions();
  create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
  create_info.ppEnabledExtensionNames = device_extensions.data();

  if (validation::enable_validation_layers)
  {
    create_info.enabledLayerCount = static_cast<uint32_t>(validation::layers.size());
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
  vkGetDeviceQueue(device, graphics_queue_index, 0, &graphics_queue);
  vkGetDeviceQueue(device, present_queue_index, 0, &present_queue);

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
  VkDescriptorSetLayoutBinding ubo_layout{};
  ubo_layout.binding            = 0;
  ubo_layout.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  ubo_layout.descriptorCount    = 1; // number of values in the UBO array
  ubo_layout.stageFlags         = VK_SHADER_STAGE_VERTEX_BIT;
  ubo_layout.pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutBinding extent_layout{};
  extent_layout.binding            = 1;
  extent_layout.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  extent_layout.descriptorCount    = 1;
  extent_layout.stageFlags         = VK_SHADER_STAGE_VERTEX_BIT;
  extent_layout.pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutBinding sampler_layout{};
  sampler_layout.binding            = 2;
  sampler_layout.descriptorCount    = 1;
  sampler_layout.descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  sampler_layout.pImmutableSamplers = nullptr;
  sampler_layout.stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT;

  std::array bindings{ubo_layout, extent_layout, sampler_layout};

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
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
