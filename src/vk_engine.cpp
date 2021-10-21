#include "cudaview/vk_engine.hpp"
#include "io.hpp"
#include "validation.hpp"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#include <algorithm> // std::min/max
#include <cstring> // memcpy
#include <iostream> // std::cerr
#include <limits> // std::numeric_limits
#include <set> // std::set
#include <stdexcept> // std::throw

/*const std::vector<Vertex> vertices = {
  { {-.5f, -.5f}, {1.f, 0.f, 0.f} },
  { { .5f, -.5f}, {0.f, 1.f, 0.f} },
  { { .5f,  .5f}, {0.f, 0.f, 1.f} },
  { {-.5f,  .5f}, {1.f, 1.f, 1.f} }
};*/
const std::vector<uint16_t> indices = { 0, 1, 2, 2, 3, 0 };
static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;

static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
{
  auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
  app->should_resize = true;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(
  const std::vector<VkSurfaceFormatKHR>& available_formats)
{
  for (const auto& available_format : available_formats)
  {
    if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
    {
      return available_format;
    }
  }
  return available_formats[0];
}

VkPresentModeKHR chooseSwapPresentMode(
  const std::vector<VkPresentModeKHR>& available_modes)
{
  VkPresentModeKHR best_mode = VK_PRESENT_MODE_FIFO_KHR;
  for (const auto& available_mode : available_modes)
  {
    if (available_mode == VK_PRESENT_MODE_MAILBOX_KHR)
    {
      return available_mode;
    }
    else if (available_mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
    {
      best_mode = available_mode;
    }
  }
  return best_mode;
}

VulkanEngine::VulkanEngine(size_t data_size):
  should_resize(false),
  element_count(data_size),
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

  staging_buffer(VK_NULL_HANDLE),
  staging_memory(VK_NULL_HANDLE),
  texture_image(VK_NULL_HANDLE),
  texture_memory(VK_NULL_HANDLE),
  texture_view(VK_NULL_HANDLE),
  texture_sampler(VK_NULL_HANDLE)
{}

VulkanEngine::VulkanEngine(): VulkanEngine(0)
{}

VulkanEngine::~VulkanEngine()
{
  cleanupSwapchain();

  ImGui_ImplVulkan_Shutdown();
  if (imgui_pool != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorPool(device, imgui_pool, nullptr);
  }
  if (texture_sampler != VK_NULL_HANDLE)
  {
    vkDestroySampler(device, texture_sampler, nullptr);
  }
  if (texture_view != VK_NULL_HANDLE)
  {
    vkDestroyImageView(device, texture_view, nullptr);
  }
  if (texture_image != VK_NULL_HANDLE)
  {
    vkDestroyImage(device, texture_image, nullptr);
  }
  if (texture_memory != VK_NULL_HANDLE)
  {
    vkFreeMemory(device, texture_memory, nullptr);
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

void VulkanEngine::mainLoop()
{
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    //ImGui::ShowDemoWindow();
    ImGui::Render();

    drawFrame();
  }
  vkDeviceWaitIdle(device);
}

void VulkanEngine::drawFrame()
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

  auto frame_idx = current_frame % MAX_FRAMES_IN_FLIGHT;
  vkWaitForFences(device, 1, &inflight_fences[frame_idx], VK_TRUE, timeout);

  // Acquire image from swap chain
  uint32_t image_idx;
  // TODO: vk_presentation_semaphore instead of image_available[frame_idx]
  auto result = vkAcquireNextImageKHR(device, swapchain, timeout,
    image_available[frame_idx], VK_NULL_HANDLE, &image_idx
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
  images_inflight[image_idx] = inflight_fences[frame_idx];

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  begin_info.pInheritanceInfo = nullptr;

  validation::checkVulkan(vkBeginCommandBuffer(
    command_buffers[image_idx], &begin_info)
  );

  VkRenderPassBeginInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  render_pass_info.renderPass = render_pass;
  render_pass_info.framebuffer = framebuffers[image_idx];
  render_pass_info.renderArea.offset = {0, 0};
  render_pass_info.renderArea.extent = swapchain_extent;

  VkClearValue clear_color = {{{.5f, .5f, .5f, 1.f}}};
  render_pass_info.clearValueCount = 1;
  render_pass_info.pClearValues = &clear_color;

  vkCmdBeginRenderPass(command_buffers[image_idx], &render_pass_info,
    VK_SUBPASS_CONTENTS_INLINE
  );
  // Note: Second parameter can be also used to bind a compute pipeline
  vkCmdBindPipeline(command_buffers[image_idx], VK_PIPELINE_BIND_POINT_GRAPHICS,
    graphics_pipeline
  );

  vkCmdBindDescriptorSets(command_buffers[image_idx],
    VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
    &descriptor_sets[image_idx], 0, nullptr
  );

  setUnstructuredRendering(command_buffers[image_idx], element_count);

  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffers[image_idx]);

  // End render pass and finish recording the command buffer
  vkCmdEndRenderPass(command_buffers[image_idx]);
  validation::checkVulkan(vkEndCommandBuffer(command_buffers[image_idx]));

  updateUniformBuffer(image_idx);

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  std::vector<VkSemaphore> wait_semaphores;
  std::vector<VkPipelineStageFlags> wait_stages;
  wait_semaphores.push_back(image_available[frame_idx]); //vk_timeline_semaphore
  wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
  getWaitFrameSemaphores(wait_semaphores, wait_stages);

  submit_info.waitSemaphoreCount = (uint32_t)wait_semaphores.size();
  submit_info.pWaitSemaphores = wait_semaphores.data();
  submit_info.pWaitDstStageMask = wait_stages.data();

  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffers[image_idx];

  std::vector<VkSemaphore> signal_semaphores;
  getSignalFrameSemaphores(signal_semaphores);
  signal_semaphores.push_back(render_finished[frame_idx]); // vk_timeline_semaphore
  submit_info.signalSemaphoreCount = (uint32_t)signal_semaphores.size();
  submit_info.pSignalSemaphores = signal_semaphores.data();

  /*VkTimelineSemaphoreSubmitInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timeline_info.waitSemaphoreValueCount = 1;
  timeline_info.pWaitSemaphoreValues = &wait_value;
  timeline_info.signalSemaphoreValueCount = 1;
  timeline_info.pSignalSemaphoreValues = &signal_value;
  submit_info.pNext = &timeline_info;*/

  vkResetFences(device, 1, &inflight_fences[frame_idx]);

  // Execute command buffer using image as attachment in framebuffer
  validation::checkVulkan(vkQueueSubmit(
    graphics_queue, 1, &submit_info, inflight_fences[frame_idx]) //VK_NULL_HANDLE
  );

  // Return image result back to swapchain for presentation on screen
  VkSwapchainKHR swapchains[] = { swapchain };
  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  //present_info.pWaitSemaphores = &vk_presentation_semaphore;
  present_info.pWaitSemaphores = &render_finished[frame_idx];
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swapchains;
  present_info.pImageIndices = &image_idx;

  result = vkQueuePresentKHR(present_queue, &present_info);
  // Resize should be done after presentation to ensure semaphore consistency
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || should_resize)
  {
    recreateSwapchain();
    should_resize = false;
  }

  current_frame++;
}

void VulkanEngine::cleanupSwapchain()
{
  for (size_t i = 0; i < swapchain_images.size(); ++i)
  {
    vkDestroyBuffer(device, uniform_buffers[i], nullptr);
    vkFreeMemory(device, ubo_memory[i], nullptr);
  }
  vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  vkFreeCommandBuffers(device, command_pool,
    static_cast<uint32_t>(command_buffers.size()), command_buffers.data()
  );
  vkDestroyPipeline(device, graphics_pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  for (auto framebuffer : framebuffers)
  {
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  }
  vkDestroyRenderPass(device, render_pass, nullptr);
  for (auto image_view : swapchain_views)
  {
    vkDestroyImageView(device, image_view, nullptr);
  }
  vkDestroySwapchainKHR(device, swapchain, nullptr);
}

void VulkanEngine::recreateSwapchain()
{
  vkDeviceWaitIdle(device);

  cleanupSwapchain();

  createSwapChain();
  createImageViews();
  createRenderPass();
  createGraphicsPipeline();
  createFramebuffers();
  createUniformBuffers();
  createDescriptorPool();
  createDescriptorSets();
  createCommandBuffers();
}

void VulkanEngine::initVulkan()
{
  createInstance();
  setupDebugMessenger();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createSwapChain();
  createImageViews();
  createRenderPass();
  createDescriptorSetLayout();
  createGraphicsPipeline();
  createFramebuffers();

  initApplication();

  createCommandPool(); // after framebuffers were created
  initImgui(); // After command pool is created
  createTextureImage();
  createTextureImageView();
  createTextureSampler();
  createVertexBuffer();
  createIndexBuffer();

  createUniformBuffers();
  createDescriptorPool();
  createDescriptorSets();

  createCommandBuffers();
  createSyncObjects();
}

void VulkanEngine::createInstance()
{
  if (validation::enable_validation_layers && !validation::checkValidationLayerSupport())
  {
    throw std::runtime_error("validation layers requested, but not supported");
  }

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Vulkan test";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "No engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_2;

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

  // TODO: Move this to some auxiliary function
  /*uint32_t extension_count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> available_exts(extension_count);
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, available_exts.data());

  std::cout << "Available extensions:\n";
  for (const auto& extension : available_exts)
  {
    std::cout << '\t' << extension.extensionName << '\n';
  }*/

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
    queue_create_info.queueCount = 1;
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
  create_info.pQueueCreateInfos = queue_create_infos.data();
  create_info.pEnabledFeatures = &device_features;
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
  validation::checkVulkan(glfwCreateWindowSurface(instance, window, nullptr, &surface));
}

VkExtent2D VulkanEngine::chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
{
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
  {
    return capabilities.currentExtent;
  }
  else
  {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    VkExtent2D actual_extent = {
      static_cast<uint32_t>(width), static_cast<uint32_t>(height)
    };
    actual_extent.width = std::clamp(actual_extent.width,
      capabilities.minImageExtent.width, capabilities.maxImageExtent.width
    );
    actual_extent.height = std::clamp(actual_extent.height,
      capabilities.minImageExtent.height, capabilities.maxImageExtent.height
    );
    return actual_extent;
  }
}

void VulkanEngine::createSwapChain()
{
  auto swapchain_support = getSwapchainProperties(physical_device);
  auto surface_format = chooseSwapSurfaceFormat(swapchain_support.formats);
  auto present_mode = chooseSwapPresentMode(swapchain_support.present_modes);
  auto extent = chooseSwapExtent(swapchain_support.capabilities);

  auto image_count = swapchain_support.capabilities.minImageCount + 1;
  const auto max_image_count = swapchain_support.capabilities.maxImageCount;
  if (max_image_count > 0 && image_count > max_image_count)
  {
    image_count = max_image_count;
  }

  VkSwapchainCreateInfoKHR create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  create_info.surface = surface;
  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  uint32_t queue_indices[2];
  findQueueFamilies(physical_device, queue_indices[0], queue_indices[1]);

  if (queue_indices[0] != queue_indices[1])
  {
    create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = queue_indices;
  }
  else
  {
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = 0;
    create_info.pQueueFamilyIndices = nullptr;
  }
  create_info.preTransform = swapchain_support.capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  create_info.presentMode = present_mode;
  create_info.clipped = VK_TRUE;
  create_info.oldSwapchain = VK_NULL_HANDLE;

  validation::checkVulkan(vkCreateSwapchainKHR(
    device, &create_info, nullptr, &swapchain)
  );

  vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr);
  swapchain_images.resize(image_count);
  vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.data());

  swapchain_format = surface_format.format;
  swapchain_extent = extent;
}

// Take buffer with shader bytecode and create a shader module from it
VkShaderModule VulkanEngine::createShaderModule(const std::vector<char>& code)
{
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size();
  create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule module;
  validation::checkVulkan(vkCreateShaderModule(device, &create_info, nullptr, &module));

  return module;
}

void VulkanEngine::createGraphicsPipeline()
{
  auto vert_code = io::readFile("_out/shaders/vertex.spv");
  auto vert_module = createShaderModule(vert_code);

  auto frag_code = io::readFile("_out/shaders/fragment.spv");
  auto frag_module = createShaderModule(frag_code);

  VkPipelineShaderStageCreateInfo vert_info{};
  vert_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vert_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vert_info.module = vert_module;
  vert_info.pName = "main"; // Entrypoint
  // Used to specify values for shader constants
  vert_info.pSpecializationInfo = nullptr;

  VkPipelineShaderStageCreateInfo frag_info{};
  frag_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  frag_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  frag_info.module = frag_module;
  frag_info.pName = "main"; // Entrypoint
  // Used to specify values for shader constants
  frag_info.pSpecializationInfo = nullptr;

  std::vector<VkVertexInputBindingDescription> bind_desc;
  std::vector<VkVertexInputAttributeDescription> attr_desc;
  getVertexDescriptions(bind_desc, attr_desc);

  VkPipelineVertexInputStateCreateInfo vert_input_info{};
  vert_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vert_input_info.vertexBindingDescriptionCount = (uint32_t)bind_desc.size();
  vert_input_info.pVertexBindingDescriptions = bind_desc.data();
  vert_input_info.vertexAttributeDescriptionCount = (uint32_t)attr_desc.size();
  vert_input_info.pVertexAttributeDescriptions = attr_desc.data();

  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  getAssemblyStateInfo(input_assembly);

  VkViewport viewport{};
  viewport.x = 0.f;
  viewport.y = 0.f;
  viewport.width = static_cast<float>(swapchain_extent.width);
  viewport.height = static_cast<float>(swapchain_extent.height);
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = swapchain_extent;

  // Combine viewport and scissor rectangle into a viewport state
  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = &viewport;
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.f;
  rasterizer.depthBiasClamp = 0.f;
  rasterizer.depthBiasSlopeFactor = 0.f;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading = 1.f;
  multisampling.pSampleMask = nullptr;
  multisampling.alphaToCoverageEnable = VK_FALSE;
  multisampling.alphaToOneEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState color_blend_attachment{};
  color_blend_attachment.colorWriteMask =
    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_blend_attachment.blendEnable = VK_FALSE;
  color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
  color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

  VkPipelineColorBlendStateCreateInfo color_blending{};
  color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.logicOp = VK_LOGIC_OP_COPY;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = &color_blend_attachment;
  color_blending.blendConstants[0] = 0.f;
  color_blending.blendConstants[1] = 0.f;
  color_blending.blendConstants[2] = 0.f;
  color_blending.blendConstants[3] = 0.f;

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &descriptor_layout;
  pipeline_layout_info.pushConstantRangeCount = 0;
  pipeline_layout_info.pPushConstantRanges = nullptr;

  validation::checkVulkan(vkCreatePipelineLayout(device, &pipeline_layout_info,
    nullptr, &pipeline_layout)
  );

  std::vector<VkPipelineShaderStageCreateInfo> stage_infos = {vert_info, frag_info};
  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = static_cast<uint32_t>(stage_infos.size());
  pipeline_info.pStages = stage_infos.data();
  pipeline_info.pVertexInputState = &vert_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pDepthStencilState = nullptr;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.pDynamicState = nullptr;
  pipeline_info.layout = pipeline_layout;
  pipeline_info.renderPass = render_pass;
  pipeline_info.subpass = 0;
  pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
  pipeline_info.basePipelineIndex = -1;

  validation::checkVulkan(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
    &pipeline_info, nullptr, &graphics_pipeline)
  );

  vkDestroyShaderModule(device, vert_module, nullptr);
  vkDestroyShaderModule(device, frag_module, nullptr);
}

void VulkanEngine::createRenderPass()
{
  VkAttachmentDescription color_attachment{};
  color_attachment.format = swapchain_format;
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference color_attachment_ref{};
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;

  // Specify memory and execution dependencies between subpasses
  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo renderpass_info{};
  renderpass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderpass_info.attachmentCount = 1;
  renderpass_info.pAttachments = &color_attachment;
  renderpass_info.subpassCount = 1;
  renderpass_info.pSubpasses = &subpass;
  renderpass_info.dependencyCount = 1;
  renderpass_info.pDependencies = &dependency;

  validation::checkVulkan(vkCreateRenderPass(
    device, &renderpass_info, nullptr, &render_pass)
  );
}

void VulkanEngine::createFramebuffers()
{
  framebuffers.resize(swapchain_views.size());
  for (size_t i = 0; i < swapchain_views.size(); ++i)
  {
    VkImageView attachments[] = { swapchain_views[i] };
    VkFramebufferCreateInfo fb_info{};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = render_pass;
    fb_info.attachmentCount = 1;
    fb_info.pAttachments = attachments;
    fb_info.width = swapchain_extent.width;
    fb_info.height = swapchain_extent.height;
    fb_info.layers = 1;

    validation::checkVulkan(vkCreateFramebuffer(
      device, &fb_info, nullptr, &framebuffers[i])
    );
  }
}

void VulkanEngine::createCommandPool()
{
  uint32_t graphics_index, present_index;
  findQueueFamilies(physical_device, graphics_index, present_index);
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = graphics_index;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  validation::checkVulkan(vkCreateCommandPool(
    device, &pool_info, nullptr, &command_pool)
  );
}

void VulkanEngine::createCommandBuffers()
{
  command_buffers.resize(framebuffers.size());

  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = command_pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = static_cast<uint32_t>(command_buffers.size());

  validation::checkVulkan(vkAllocateCommandBuffers(
    device, &alloc_info, command_buffers.data())
  );
}

void VulkanEngine::setUnstructuredRendering(VkCommandBuffer& cmd_buffer,
  uint32_t vertex_count)
{
  VkBuffer vertex_buffers[] = { vertex_buffer };
  VkDeviceSize offsets[] = { 0 };
  auto binding_count = sizeof(vertex_buffers) / sizeof(vertex_buffers[0]);
  vkCmdBindVertexBuffers(cmd_buffer, 0, binding_count, vertex_buffers, offsets);
  vkCmdDraw(cmd_buffer, vertex_count, 1, 0, 0);
  // NOTE: For indexed drawing, use the following:
  //vkCmdBindIndexBuffer(command_buffers[i], index_buffer, 0, VK_INDEX_TYPE_UINT16);
  //auto index_count = static_cast<uint32_t>(indices.size());
  //vkCmdDrawIndexed(command_buffers[i], index_count, 1, 0, 0, 0);*/
}

uint32_t findMemoryType(VkPhysicalDevice ph_device, uint32_t type_filter,
  VkMemoryPropertyFlags properties)
{
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(ph_device, &mem_props);

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
  {
    if ((type_filter & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & properties) == properties)
    {
      return i;
    }
  }
  return ~0;
}

void VulkanEngine::createSyncObjects()
{
  images_inflight.resize(swapchain_images.size(), VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  /*validation::checkVulkan(vkCreateSemaphore(
    device, &semaphore_info, nullptr, &vk_presentation_semaphore)
  );*/
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
  {
    validation::checkVulkan(vkCreateSemaphore(
      device, &semaphore_info, nullptr, &image_available[i])
    );
    validation::checkVulkan(vkCreateSemaphore(
      device, &semaphore_info, nullptr, &render_finished[i])
    );
    validation::checkVulkan(vkCreateFence(
      device, &fence_info, nullptr, &inflight_fences[i])
    );
  }
}

void VulkanEngine::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
  VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory &memory)
{
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  validation::checkVulkan(vkCreateBuffer(device, &buffer_info, nullptr, &buffer));

  VkMemoryRequirements mem_req;
  vkGetBufferMemoryRequirements(device, buffer, &mem_req);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_req.size;
  auto type = findMemoryType(physical_device, mem_req.memoryTypeBits, properties);
  alloc_info.memoryTypeIndex = type;
  validation::checkVulkan(vkAllocateMemory(device, &alloc_info, nullptr, &memory));

  vkBindBufferMemory(device, buffer, memory, 0);
}

void VulkanEngine::createVertexBuffer()
{
  VkDeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();

  VkBuffer staging_buffer;
  VkDeviceMemory staging_buffer_memory;
  createBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    staging_buffer, staging_buffer_memory
  );

  void *data;
  vkMapMemory(device, staging_buffer_memory, 0, buffer_size, 0, &data);
  memcpy(data, vertices.data(), (size_t) buffer_size);
  vkUnmapMemory(device, staging_buffer_memory);

  createBuffer(buffer_size,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer, vertex_buffer_memory
  );
  copyBuffer(staging_buffer, vertex_buffer, buffer_size);

  vkDestroyBuffer(device, staging_buffer, nullptr);
  vkFreeMemory(device, staging_buffer_memory, nullptr);
}

void VulkanEngine::createIndexBuffer()
{
  VkDeviceSize buffer_size = sizeof(indices[0]) * indices.size();
  VkBuffer staging_buffer;
  VkDeviceMemory staging_buffer_memory;
  createBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    staging_buffer, staging_buffer_memory
  );

  void *data;
  vkMapMemory(device, staging_buffer_memory, 0, buffer_size, 0, &data);
  memcpy(data, indices.data(), (size_t) buffer_size);
  vkUnmapMemory(device, staging_buffer_memory);

  createBuffer(buffer_size,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, index_buffer, index_buffer_memory
  );
  copyBuffer(staging_buffer, index_buffer, buffer_size);

  vkDestroyBuffer(device, staging_buffer, nullptr);
  vkFreeMemory(device, staging_buffer_memory, nullptr);
}

void VulkanEngine::createExternalBuffer(VkDeviceSize size,
  VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
  VkExternalMemoryHandleTypeFlagsKHR handle_type, VkBuffer& buffer,
  VkDeviceMemory& buffer_memory)
{
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkExternalMemoryBufferCreateInfo extmem_info{};
  extmem_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  extmem_info.handleTypes = handle_type;
  buffer_info.pNext = &extmem_info;

  validation::checkVulkan(vkCreateBuffer(device, &buffer_info, nullptr, &buffer));

  VkMemoryRequirements mem_reqs;
  vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);
  VkExportMemoryAllocateInfoKHR export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
  export_info.pNext = nullptr;
  export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.pNext = &export_info;
  alloc_info.allocationSize = mem_reqs.size;
  auto type = findMemoryType(physical_device, mem_reqs.memoryTypeBits, properties);
  alloc_info.memoryTypeIndex = type;

  validation::checkVulkan(vkAllocateMemory(
    device, &alloc_info, nullptr, &buffer_memory)
  );
  vkBindBufferMemory(device, buffer, buffer_memory, 0);
}

void *VulkanEngine::getMemHandle(VkDeviceMemory memory,
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

void *VulkanEngine::getSemaphoreHandle(VkSemaphore semaphore,
  VkExternalSemaphoreHandleTypeFlagBits handle_type)
{
  int fd;
  VkSemaphoreGetFdInfoKHR fd_info{};
  fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  fd_info.pNext = nullptr;
  fd_info.semaphore = semaphore;
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

void VulkanEngine::createExternalSemaphore(VkSemaphore& semaphore)
{
  /*VkSemaphoreTypeCreateInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_info.pNext = nullptr;
  timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_info.initialValue = 0;*/

  VkExportSemaphoreCreateInfoKHR export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
  export_info.pNext = nullptr;
  export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
  export_info.pNext = nullptr;
  //export_info.pNext = &timeline_info;

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_info.pNext = &export_info;

  validation::checkVulkan(
    vkCreateSemaphore(device, &semaphore_info, nullptr, &semaphore)
  );
}

void VulkanEngine::importExternalBuffer(void *handle, size_t size,
  VkExternalMemoryHandleTypeFlagBits handle_type, VkBufferUsageFlags usage,
  VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory)
{
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  validation::checkVulkan(vkCreateBuffer(device, &buffer_info, nullptr, &buffer));

  VkMemoryRequirements mem_req;
  vkGetBufferMemoryRequirements(device, buffer, &mem_req);

  VkImportMemoryFdInfoKHR handle_info{};
  handle_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
  handle_info.pNext = nullptr;
  handle_info.fd = (int)(uintptr_t)handle;
  handle_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkMemoryAllocateInfo mem_alloc{};
  mem_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mem_alloc.pNext = (void*)&handle_info;
  mem_alloc.allocationSize = size;
  auto type = findMemoryType(physical_device, mem_req.memoryTypeBits, properties);
  mem_alloc.memoryTypeIndex = type;

  validation::checkVulkan(vkAllocateMemory(device, &mem_alloc, nullptr, &memory));
  vkBindBufferMemory(device, buffer, memory, 0);
}

// Return list of required GLFW extensions and additional required validation layers
std::vector<const char*> VulkanEngine::getRequiredExtensions() const
{
  uint32_t glfw_ext_count = 0;
  const char **glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
  std::vector<const char*> extensions(glfw_exts, glfw_exts + glfw_ext_count);
  if (validation::enable_validation_layers)
  {
    // Enable debugging message extension
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  return extensions;
}

std::vector<const char*> VulkanEngine::getRequiredDeviceExtensions() const
{
  std::vector<const char*> extensions;
  extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  return extensions;
}

bool VulkanEngine::checkAllExtensionsSupported(VkPhysicalDevice dev,
  const std::vector<const char*>& device_extensions) const
{
  // Enumerate extensions and check if all required extensions are included
  uint32_t ext_count;
  vkEnumerateDeviceExtensionProperties(dev, nullptr, &ext_count, nullptr);
  std::vector<VkExtensionProperties> available_extensions(ext_count);
  vkEnumerateDeviceExtensionProperties(dev, nullptr, &ext_count,
    available_extensions.data()
  );

  std::set<std::string> required_extensions(
    device_extensions.begin(), device_extensions.end()
  );

  for (const auto& extension : available_extensions)
  {
    required_extensions.erase(extension.extensionName);
  }
  return required_extensions.empty();
}

bool VulkanEngine::isDeviceSuitable(VkPhysicalDevice dev) const
{
  uint32_t graphics_index, present_index;
  auto has_queues = findQueueFamilies(dev, graphics_index, present_index);
  auto device_extensions = getRequiredDeviceExtensions();
  auto supports_extensions = checkAllExtensionsSupported(dev, device_extensions);
  auto swapchain_support = getSwapchainProperties(dev);
  auto swapchain_adequate = !swapchain_support.formats.empty() &&
                            !swapchain_support.present_modes.empty();
  VkPhysicalDeviceFeatures supported_features;
  vkGetPhysicalDeviceFeatures(dev, &supported_features);
  return supports_extensions && swapchain_adequate && has_queues
    && supported_features.samplerAnisotropy;
}

// Logic to find queue family indices to populate struct with
bool VulkanEngine::findQueueFamilies(VkPhysicalDevice dev,
  uint32_t& graphics_family, uint32_t& present_family) const
{
  constexpr auto family_empty = ~0u;
  // Assign index to queue families that could be found
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_family_count,
    queue_families.data()
  );

  graphics_family = present_family = family_empty;

  // Find at least one queue family that supports VK_QUEUE_GRAPHICS_BIT
  for (uint32_t i = 0; i < queue_family_count; ++i)
  {
    if (queue_families[i].queueCount > 0)
    {
      if (graphics_family == family_empty && queue_families[i].queueFlags &
          VK_QUEUE_GRAPHICS_BIT)
      {
        graphics_family = i;
      }
      uint32_t present_support = 0;
      vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &present_support);
      if (present_family == family_empty && present_support)
      {
        present_family = i;
      }
      if (present_family != family_empty && graphics_family != family_empty)
      {
        break;
      }
    }
  }
  return graphics_family != family_empty && present_family != family_empty;
}

SwapChainSupportDetails VulkanEngine::getSwapchainProperties(
  VkPhysicalDevice dev) const
{
  SwapChainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface, &details.capabilities);

  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &format_count, nullptr);
  if (format_count != 0)
  {
    details.formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &format_count,
      details.formats.data()
    );
  }

  uint32_t mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &mode_count, nullptr);
  if (mode_count != 0)
  {
    details.present_modes.resize(mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &mode_count,
      details.present_modes.data()
    );
  }
  return details;
}

void VulkanEngine::getVertexDescriptions(
  std::vector<VkVertexInputBindingDescription>& bind_desc,
  std::vector<VkVertexInputAttributeDescription>& attr_desc)
{
  bind_desc.resize(1);
  bind_desc[0].binding = 0;
  bind_desc[0].stride = sizeof(Vertex);
  bind_desc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  attr_desc.resize(2);
  attr_desc[0].binding = 0;
  attr_desc[0].location = 0;
  attr_desc[0].format = VK_FORMAT_R32G32_SFLOAT;
  attr_desc[0].offset = offsetof(Vertex, pos);
  attr_desc[1].binding = 0;
  attr_desc[1].location = 1;
  attr_desc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attr_desc[1].offset = offsetof(Vertex, color);
}

void VulkanEngine::getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info)
{
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  //info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  info.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  info.primitiveRestartEnable = VK_FALSE;
}

void VulkanEngine::initApplication()
{
  //createExternalSemaphore(vk_timeline_semaphore);
  createExternalSemaphore(vk_wait_semaphore);
  createExternalSemaphore(vk_signal_semaphore);
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
  pool_info.maxSets = 1000;
  pool_info.poolSizeCount = std::size(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;

  validation::checkVulkan(vkCreateDescriptorPool(
    device, &pool_info, nullptr, &imgui_pool)
  );

  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForVulkan(window, true);

  ImGui_ImplVulkan_InitInfo init_info{};
  init_info.Instance = instance;
  init_info.PhysicalDevice = physical_device;
  init_info.Device = device;
  init_info.Queue = graphics_queue;
  init_info.DescriptorPool = imgui_pool;
  init_info.MinImageCount = 3; // TODO: Check if this is true
  init_info.ImageCount = 3;
  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  ImGui_ImplVulkan_Init(&init_info, render_pass);

  auto cmd = beginSingleTimeCommands();
  ImGui_ImplVulkan_CreateFontsTexture(cmd);
  endSingleTimeCommands(cmd);
  ImGui_ImplVulkan_DestroyFontUploadObjects();
}

VkCommandBuffer VulkanEngine::beginSingleTimeCommands()
{
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandPool = command_pool;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer command_buffer;
  vkAllocateCommandBuffers(device, &alloc_info, &command_buffer);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(command_buffer, &begin_info);
  return command_buffer;
}

void VulkanEngine::endSingleTimeCommands(VkCommandBuffer command_buffer)
{
  // Finish recording the command buffer
  vkEndCommandBuffer(command_buffer);

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue);
  vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}

void VulkanEngine::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
{
  // Memory transfers are commands executed with buffers, just like drawing
  auto cmd_buffer = beginSingleTimeCommands();

  VkBufferCopy copy_region{};
  copy_region.srcOffset = 0;
  copy_region.dstOffset = 0;
  copy_region.size = size;
  vkCmdCopyBuffer(cmd_buffer, src, dst, 1, &copy_region);

  endSingleTimeCommands(cmd_buffer);
}

void VulkanEngine::getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
  std::vector<VkPipelineStageFlags>& wait_stages) const
{}

void VulkanEngine::getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const
{}

void VulkanEngine::createDescriptorSetLayout()
{
  VkDescriptorSetLayoutBinding ubo_layout{};
  ubo_layout.binding = 0;
  ubo_layout.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  ubo_layout.descriptorCount = 1; // number of values in the UBO array
  ubo_layout.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  ubo_layout.pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutBinding sampler_layout{};
  sampler_layout.binding = 1;
  sampler_layout.descriptorCount = 1;
  sampler_layout.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  sampler_layout.pImmutableSamplers = nullptr;
  sampler_layout.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  std::array bindings{ubo_layout, sampler_layout};

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();

  validation::checkVulkan(vkCreateDescriptorSetLayout(
    device, &layout_info, nullptr, &descriptor_layout)
  );
}

void VulkanEngine::createUniformBuffers()
{
  VkDeviceSize buffer_size = sizeof(UniformBufferObject);
  auto img_count = swapchain_images.size();
  uniform_buffers.resize(img_count);
  ubo_memory.resize(img_count);

  for (size_t i = 0; i < img_count; ++i)
  {
    createBuffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      uniform_buffers[i], ubo_memory[i]
    );
  }
}

void VulkanEngine::updateUniformBuffer(uint32_t image_index)
{
  UniformBufferObject ubo{};
  ubo.model = glm::mat4(1.f);
  ubo.view = glm::mat4(1.f);
  ubo.proj = glm::mat4(1.f);
  /*ubo.model =
  ubo.view =
  auto aspect_ratio = swapchain_extent.width / (float)swapchain_extent.height;
  ubo.proj = glm::perspective(glm::radians(45.f), aspect_ratio, .1f, 10.f);
  ubo.proj[1][1] *= -1;*/

  void *data = nullptr;
  vkMapMemory(device, ubo_memory[image_index], 0, sizeof(ubo), 0, &data);
  memcpy(data, &ubo, sizeof(ubo));
  vkUnmapMemory(device, ubo_memory[image_index]);
}

void VulkanEngine::createDescriptorPool()
{
  std::array<VkDescriptorPoolSize, 2> pool_sizes{};
  pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  pool_sizes[0].descriptorCount = static_cast<uint32_t>(swapchain_images.size());
  pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  pool_sizes[1].descriptorCount = static_cast<uint32_t>(swapchain_images.size());

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
  pool_info.pPoolSizes = pool_sizes.data();
  pool_info.maxSets = static_cast<uint32_t>(swapchain_images.size());
  pool_info.flags = 0;

  validation::checkVulkan(
    vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool)
  );
}

void VulkanEngine::createDescriptorSets()
{
  auto img_count = swapchain_images.size();
  std::vector<VkDescriptorSetLayout> layouts(img_count, descriptor_layout);
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = descriptor_pool;
  alloc_info.descriptorSetCount = static_cast<uint32_t>(img_count);
  alloc_info.pSetLayouts = layouts.data();

  descriptor_sets.resize(swapchain_images.size());
  validation::checkVulkan(
    vkAllocateDescriptorSets(device, &alloc_info, descriptor_sets.data())
  );

  for (size_t i = 0; i < img_count; ++i)
  {
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = uniform_buffers[i];
    buffer_info.offset = 0;
    buffer_info.range = sizeof(UniformBufferObject); // or VK_WHOLE_SIZE

    VkDescriptorImageInfo image_info{};
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_info.imageView = texture_view;
    image_info.sampler = texture_sampler;

    std::array<VkWriteDescriptorSet, 2> desc_writes{};
    desc_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    desc_writes[0].dstSet = descriptor_sets[i];
    desc_writes[0].dstBinding = 0;
    desc_writes[0].dstArrayElement = 0;
    desc_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    desc_writes[0].descriptorCount = 1;
    desc_writes[0].pBufferInfo = &buffer_info;
    desc_writes[0].pImageInfo = nullptr;
    desc_writes[0].pTexelBufferView = nullptr;

    desc_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    desc_writes[1].dstSet = descriptor_sets[i];
    desc_writes[1].dstBinding = 1;
    desc_writes[1].dstArrayElement = 0;
    desc_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    desc_writes[1].descriptorCount = 1;
    desc_writes[1].pImageInfo = &image_info;

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(desc_writes.size()),
      desc_writes.data(), 0, nullptr
    );
  }
}

void VulkanEngine::createImage(uint32_t width, uint32_t height, VkFormat format,
  VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
  VkImage& image, VkDeviceMemory& image_memory)
{
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.extent.width = width;
  image_info.extent.height = height;
  image_info.extent.depth = 1;
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.format = format;
  image_info.tiling = tiling;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage = usage;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.flags = 0;

  validation::checkVulkan(
    vkCreateImage(device, &image_info, nullptr, &texture_image)
  );
  VkMemoryRequirements mem_req;
  vkGetImageMemoryRequirements(device, texture_image, &mem_req);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_req.size;
  alloc_info.memoryTypeIndex =
    findMemoryType(physical_device, mem_req.memoryTypeBits, properties);

  validation::checkVulkan(
    vkAllocateMemory(device, &alloc_info, nullptr, &texture_memory)
  );
  vkBindImageMemory(device, texture_image, texture_memory, 0);
}

void VulkanEngine::createTextureImage()
{
  std::string filename = "textures/lena512.png";
  int tex_width, tex_height, tex_channels;
  auto pixels = io::loadImage(filename, tex_width, tex_height, tex_channels);
  if (!pixels)
  {
    throw std::runtime_error("failed to load texture image");
  }
  VkDeviceSize img_size = tex_width * tex_height * 4;

  createBuffer(img_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    staging_buffer, staging_memory
  );

  void *data;
  vkMapMemory(device, staging_memory, 0, img_size, 0, &data);
  memcpy(data, pixels, static_cast<size_t>(img_size));
  vkUnmapMemory(device, staging_memory);

  stbi_image_free(pixels);

  createImage(tex_width, tex_height,
    VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    texture_image, texture_memory
  );

  transitionImageLayout(texture_image, VK_FORMAT_R8G8B8A8_SRGB,
    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
  );
  copyBufferToImage(staging_buffer, texture_image, tex_width, tex_height);
  transitionImageLayout(texture_image, VK_FORMAT_R8G8B8A8_SRGB,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
  );

  vkDestroyBuffer(device, staging_buffer, nullptr);
  vkFreeMemory(device, staging_memory, nullptr);
}

void VulkanEngine::transitionImageLayout(VkImage image, VkFormat format,
  VkImageLayout old_layout, VkImageLayout new_layout)
{
  auto cmd_buffer = beginSingleTimeCommands();

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel   = 0;
  barrier.subresourceRange.levelCount     = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount     = 1;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = 0;

  VkPipelineStageFlags src_stage, dst_stage;
  if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
      new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
  {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  }
  else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
           new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
  {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }
  else
  {
    throw std::invalid_argument("unsupported layout transition");
  }

  vkCmdPipelineBarrier(cmd_buffer, src_stage, dst_stage,
    0, 0, nullptr, 0, nullptr, 1, &barrier
  );

  endSingleTimeCommands(cmd_buffer);
}

void VulkanEngine::copyBufferToImage(VkBuffer buffer, VkImage image,
  uint32_t width, uint32_t height)
{
  auto cmd_buffer = beginSingleTimeCommands();

  VkBufferImageCopy region{};
  region.bufferOffset      = 0;
  region.bufferRowLength   = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel       = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount     = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {width, height, 1};

  vkCmdCopyBufferToImage(cmd_buffer, buffer, image,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
  );

  endSingleTimeCommands(cmd_buffer);
}

VkImageView VulkanEngine::createImageView(VkImage image, VkFormat format)
{
  VkImageViewCreateInfo view_info{};
  view_info.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image    = image;
  // Treat image as 1D/2D/3D texture or as a cube map
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format   = format;
  // Default mapping of all color channels
  view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  // Describe image purpose and which part of it should be accesssed
  view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel   = 0;
  view_info.subresourceRange.levelCount     = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount     = 1;

  VkImageView image_view;
  validation::checkVulkan(
    vkCreateImageView(device, &view_info, nullptr, &image_view)
  );
  return image_view;
}

// Set up image views, so they can be used as color targets later on
void VulkanEngine::createImageViews()
{
  swapchain_views.resize(swapchain_images.size());
  // Create a basic image view for every image in the swap chain
  for (size_t i = 0; i < swapchain_images.size(); ++i)
  {
    swapchain_views[i] = createImageView(swapchain_images[i], swapchain_format);
  }
}

void VulkanEngine::createTextureImageView()
{
  texture_view = createImageView(texture_image, VK_FORMAT_R8G8B8A8_SRGB);
}

void VulkanEngine::createTextureSampler()
{
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical_device, &properties);

  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = VK_FILTER_LINEAR;
  sampler_info.minFilter = VK_FILTER_LINEAR;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.anisotropyEnable = VK_TRUE;
  sampler_info.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
  sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler_info.mipLodBias = 0.f;
  sampler_info.minLod = 0.f;
  sampler_info.maxLod = 0.f;

  validation::checkVulkan(
    vkCreateSampler(device, &sampler_info, nullptr, &texture_sampler)
  );
}
