#include "cudaview/vk_engine.hpp"
#include "io.hpp"
#include "validation.hpp"

#include <algorithm> // std::min/max
#include <cstring> // memcpy
#include <cstdint> // UINT32_MAX
#include <set> // std::set
#include <stdexcept> // std::throw

const int MAX_FRAMES_IN_FLIGHT = 2;
const std::vector<Vertex> vertices = {
  { {-.5f, -.5f}, {1.f, 0.f, 0.f} },
  { { .5f, -.5f}, {0.f, 1.f, 0.f} },
  { { .5f,  .5f}, {0.f, 0.f, 1.f} },
  { {-.5f,  .5f}, {1.f, 1.f, 1.f} }
};
const std::vector<uint16_t> indices = { 0, 1, 2, 2, 3, 0 };

static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
{
  auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
  app->should_resize = true;
}

// Return list of required GLFW extensions and additional required validation layers
std::vector<const char*> getRequiredExtensions()
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

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& available_modes)
{
  for (const auto& available_mode : available_modes)
  {
    if (available_mode == VK_PRESENT_MODE_MAILBOX_KHR)
    {
      return available_mode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
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

void VulkanEngine::run(std::function<void(void)> func, size_t step_count)
{
  for (int i = 0; i < step_count; ++i)
  {
    func();
    drawFrame();
  }
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    drawFrame();
  }

  vkDeviceWaitIdle(device);
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
  createGraphicsPipeline();
  createFramebuffers();
  createCommandPool(); // after framebuffers were created
  createVertexBuffer();
  createIndexBuffer();
  createCommandBuffers();
  createSyncObjects();
}

void VulkanEngine::cleanup()
{
  cleanupSwapchain();

  vkDestroyBuffer(device, vertex_buffer, nullptr);
  vkFreeMemory(device, vertex_buffer_memory, nullptr);
  vkDestroyBuffer(device, index_buffer, nullptr);
  vkFreeMemory(device, index_buffer_memory, nullptr);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
  {
    vkDestroySemaphore(device, render_finished[i], nullptr);
    vkDestroySemaphore(device, image_available[i], nullptr);
    vkDestroyFence(device, inflight_fences[i], nullptr);
  }
  vkDestroyCommandPool(device, command_pool, nullptr);
  vkDestroyDevice(device, nullptr);
  if (validation::enable_validation_layers)
  {
    validation::DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
  }
  // Surface must be destroyed before instance
  vkDestroySurfaceKHR(instance, surface, nullptr);
  vkDestroyInstance(instance, nullptr);
  glfwDestroyWindow(window);
  glfwTerminate();
}

void VulkanEngine::cleanupSwapchain()
{
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
  createCommandBuffers();
}

void VulkanEngine::drawFrame()
{
  // Wait for the frame to be finished
  vkWaitForFences(device, 1, &inflight_fences[current_frame], VK_TRUE, UINT64_MAX);

  // Acquire image from swap chain
  uint32_t image_idx;
  // Third parameter means timeout for acquiring image is disabled
  auto result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
    image_available[current_frame], VK_NULL_HANDLE, &image_idx
  );
  // Check if swapchain is no longer adequate for presentation (resize, etc.)
  if (result == VK_ERROR_OUT_OF_DATE_KHR)
  {
    recreateSwapchain();
    return;
  }
  else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
  {
    throw std::runtime_error("failed to acquire swapchain image!");
  }

  // Check if a previous frame is using this image (have to wait for its fence)
  if (images_inflight[image_idx] != VK_NULL_HANDLE)
  {
    vkWaitForFences(device, 1, &images_inflight[image_idx], VK_TRUE, UINT64_MAX);
  }
  // Mark the image as now being in use by this frame
  images_inflight[image_idx] = inflight_fences[current_frame];

  // Submit the command buffer
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  VkSemaphore wait_semaphores[] = { image_available[current_frame] };
  VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = wait_semaphores;
  submit_info.pWaitDstStageMask = wait_stages;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffers[image_idx];

  VkSemaphore signal_semaphores[] = { render_finished[current_frame] };
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = signal_semaphores;

  // Fences must be reset before being submitted again
  vkResetFences(device, 1, &inflight_fences[current_frame]);

  // Execute command buffer using image as attachment in framebuffer
  validation::checkVulkan(vkQueueSubmit(graphics_queue, 1, &submit_info,
    inflight_fences[current_frame])
  );

  // Return image result back to swapchain for presentation on screen
  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = signal_semaphores;

  VkSwapchainKHR swapchains[] = { swapchain };
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swapchains;
  present_info.pImageIndices = &image_idx;
  // Used to check each swapchain if presentation is successful
  present_info.pResults = nullptr;

  result = vkQueuePresentKHR(present_queue, &present_info);
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || should_resize)
  {
    // Resize should be done after presenting to ensure that semaphores are
    // in a consistent state
    should_resize = false;
    recreateSwapchain();
  }
  else if (result != VK_SUCCESS)
  {
    throw std::runtime_error("failed to present swapchain image!");
  }

  // Wait for work to finish right after submitting it
  vkQueueWaitIdle(present_queue);
  current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanEngine::createInstance()
{
  if (validation::enable_validation_layers && !validation::checkValidationLayerSupport())
  {
    throw std::runtime_error("validation layers requested, but not available");
  }

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Vulkan test";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "No engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  auto glfw_extensions = getRequiredExtensions();
  create_info.enabledExtensionCount = static_cast<uint32_t>(glfw_extensions.size());
  create_info.ppEnabledExtensionNames = glfw_extensions.data();

  VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
  // Include validation layer names if they are enabled
  if (validation::enable_validation_layers)
  {
    create_info.enabledLayerCount = static_cast<uint32_t>(validation::layers.size());
    create_info.ppEnabledLayerNames = validation::layers.data();

    validation::populateDebugMessengerCreateInfo(debug_create_info);
    create_info.pNext = static_cast<VkDebugUtilsMessengerCreateInfoEXT*>(
      &debug_create_info
    );
  }
  else
  {
    create_info.enabledLayerCount = 0;
    create_info.pNext = nullptr;
  }

  uint32_t extension_count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> extensions(extension_count);
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data());

  std::cout << "Available extensions:\n";
  for (const auto& extension : extensions)
  {
    std::cout << '\t' << extension.extensionName << '\n';
  }

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

  for (const auto& device : devices)
  {
    if (isDeviceSuitable(device))
    {
      physical_device = device;
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
  auto indices = findQueueFamilies(physical_device);

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::set<uint32_t> unique_queue_families =
    {indices.graphics_family.value(), indices.present_family.value()};
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

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
  create_info.pQueueCreateInfos = queue_create_infos.data();
  create_info.pEnabledFeatures = &device_features;
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

  validation::checkVulkan(vkCreateDevice(physical_device, &create_info, nullptr, &device));

  // Must be called after logical device is created (obviously!)
  vkGetDeviceQueue(device, indices.graphics_family.value(), 0, &graphics_queue);
  vkGetDeviceQueue(device, indices.present_family.value(), 0, &present_queue);
}

void VulkanEngine::createSurface()
{
  validation::checkVulkan(glfwCreateWindowSurface(instance, window, nullptr, &surface));
}

// Logic to find queue family indices to populate struct with
QueueFamilyIndices VulkanEngine::findQueueFamilies(VkPhysicalDevice device)
{
  // Assign index to queue families that could be found
  QueueFamilyIndices indices;
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
    queue_families.data()
  );

  // Find at least one queue family that supports VK_QUEUE_GRAPHICS_BIT
  int i = 0;
  for (const auto& queue_family : queue_families)
  {
    VkBool32 present_support = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
    if (present_support)
    {
      indices.present_family = i;
    }
    if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
    {
      indices.graphics_family = i;
    }
    if (indices.isComplete())
    {
      break;
    }
    i++;
  }

  return indices;

}

bool VulkanEngine::checkDeviceExtensionSupport(VkPhysicalDevice device)
{
  // Enumerate extensions and check if all required extensions are included
  uint32_t ext_count;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, nullptr);
  std::vector<VkExtensionProperties> available_extensions(ext_count);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count,
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

bool VulkanEngine::isDeviceSuitable(VkPhysicalDevice device)
{
  auto indices = findQueueFamilies(device);
  auto extensions_supported = checkDeviceExtensionSupport(device);
  bool swapchain_adequate = false;
  if (extensions_supported)
  {
    auto swapchain_support = querySwapChainSupport(device);
    swapchain_adequate = !swapchain_support.formats.empty() &&
                         !swapchain_support.present_modes.empty();
  }
  return indices.isComplete() && extensions_supported && swapchain_adequate;
}

VkExtent2D VulkanEngine::chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
{
  if (capabilities.currentExtent.width != UINT32_MAX)
  {
    return capabilities.currentExtent;
  }
  else
  {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    VkExtent2D actual_extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    actual_extent.width = std::clamp(actual_extent.width,
      capabilities.minImageExtent.width, capabilities.maxImageExtent.width
    );
    actual_extent.height = std::clamp(actual_extent.height,
      capabilities.minImageExtent.height, capabilities.maxImageExtent.height
    );
    return actual_extent;
  }
}

SwapChainSupportDetails VulkanEngine::querySwapChainSupport(VkPhysicalDevice device)
{
  SwapChainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
  if (format_count != 0)
  {
    details.formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count,
      details.formats.data()
    );
  }

  uint32_t mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count, nullptr);
  if (mode_count != 0)
  {
    details.present_modes.resize(mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &mode_count,
      details.present_modes.data()
    );
  }
  return details;
}

void VulkanEngine::createSwapChain()
{
  auto swapchain_support = querySwapChainSupport(physical_device);
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

  auto indices = findQueueFamilies(physical_device);
  uint32_t queue_family_indices[] = {
    indices.graphics_family.value(), indices.present_family.value()
  };

  if (indices.graphics_family != indices.present_family)
  {
    create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = queue_family_indices;
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

  validation::checkVulkan(vkCreateSwapchainKHR(device, &create_info, nullptr, &swapchain));

  vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr);
  swapchain_images.resize(image_count);
  vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.data());

  swapchain_format = surface_format.format;
  swapchain_extent = extent;
}

// Set up image views, so they can be used as color targets later on
void VulkanEngine::createImageViews()
{
  swapchain_views.resize(swapchain_images.size());
  // Create a basic image view for every image in the swap chain
  for (size_t i = 0; i < swapchain_images.size(); ++i)
  {
    VkImageViewCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    create_info.image = swapchain_images[i];
    // Treat image as 1D/2D/3D texture or as a cube map
    create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    create_info.format = swapchain_format;
    // Default mapping of all color channels
    create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    // Describe image purpose and which part of it should be accesssed
    create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    create_info.subresourceRange.baseMipLevel = 0;
    create_info.subresourceRange.levelCount = 1;
    create_info.subresourceRange.baseArrayLayer = 0;
    create_info.subresourceRange.layerCount = 1;

    // Create image view
    validation::checkVulkan(vkCreateImageView(
      device, &create_info, nullptr, &swapchain_views[i])
    );
  }
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

  VkPipelineShaderStageCreateInfo shader_stages[] = {vert_info, frag_info};

  auto binding_desc = Vertex::getBindingDescription();
  auto attribute_desc = Vertex::getAttributeDescriptions();

  VkPipelineVertexInputStateCreateInfo vert_input_info{};
  vert_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vert_input_info.vertexBindingDescriptionCount = 1;
  vert_input_info.pVertexBindingDescriptions = &binding_desc;
  vert_input_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_desc.size());
  vert_input_info.pVertexAttributeDescriptions = attribute_desc.data();

  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly.primitiveRestartEnable = VK_FALSE;

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
  color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
    VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
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
  pipeline_layout_info.setLayoutCount = 0;
  pipeline_layout_info.pSetLayouts = nullptr;
  pipeline_layout_info.pushConstantRangeCount = 0;
  pipeline_layout_info.pPushConstantRanges = nullptr;

  validation::checkVulkan(vkCreatePipelineLayout(device, &pipeline_layout_info,
    nullptr, &pipeline_layout)
  );

  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = 2;
  pipeline_info.pStages = shader_stages;
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
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount = 1;
  render_pass_info.pAttachments = &color_attachment;
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  render_pass_info.dependencyCount = 1;
  render_pass_info.pDependencies = &dependency;

  validation::checkVulkan(vkCreateRenderPass(
    device, &render_pass_info, nullptr, &render_pass)
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
  auto queue_family_indices = findQueueFamilies(physical_device);
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = queue_family_indices.graphics_family.value();
  pool_info.flags = 0;

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

  // Start command buffer recording
  for (size_t i = 0; i < command_buffers.size(); ++i)
  {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = 0;
    begin_info.pInheritanceInfo = nullptr;

    validation::checkVulkan(vkBeginCommandBuffer(command_buffers[i], &begin_info));

    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = render_pass;
    render_pass_info.framebuffer = framebuffers[i];
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = swapchain_extent;

    VkClearValue clear_color = {{{.5f, .5f, .5f, 1.f}}};
    render_pass_info.clearValueCount = 1;
    render_pass_info.pClearValues = &clear_color;

    vkCmdBeginRenderPass(command_buffers[i], &render_pass_info,
      VK_SUBPASS_CONTENTS_INLINE
    );

    // Bind the graphics pipeline
    // Note: Second parameter can be used to bind a compute pipeline
    vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
      graphics_pipeline
    );

    VkBuffer vertex_buffers[] = { vertex_buffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(command_buffers[i], 0, 1, vertex_buffers, offsets);

    vkCmdBindIndexBuffer(command_buffers[i], index_buffer, 0, VK_INDEX_TYPE_UINT16);

    // vkCmdDraw(...) is a draw command without index buffers
    vkCmdDrawIndexed(command_buffers[i], static_cast<uint32_t>(indices.size()),
      1, 0, 0, 0
    );

    // End render pass and finish recording the command buffer
    vkCmdEndRenderPass(command_buffers[i]);
    validation::checkVulkan(vkEndCommandBuffer(command_buffers[i]));
  }
}

void VulkanEngine::createSyncObjects()
{
  image_available.resize(MAX_FRAMES_IN_FLIGHT);
  render_finished.resize(MAX_FRAMES_IN_FLIGHT);
  inflight_fences.resize(MAX_FRAMES_IN_FLIGHT);
  images_inflight.resize(swapchain_images.size(), VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
  {
    validation::checkVulkan(
      vkCreateSemaphore(device, &semaphore_info, nullptr, &image_available[i])
    );
    validation::checkVulkan(
      vkCreateSemaphore(device, &semaphore_info, nullptr, &render_finished[i])
    );
    validation::checkVulkan(
      vkCreateFence(device, &fence_info, nullptr, &inflight_fences[i])
    );
  }
}

uint32_t VulkanEngine::findMemoryType(uint32_t type_filter,
  VkMemoryPropertyFlags properties)
{
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
  {
    if ((type_filter & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & properties) == properties)
    {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
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
  alloc_info.memoryTypeIndex = findMemoryType(mem_req.memoryTypeBits, properties);
  validation::checkVulkan(vkAllocateMemory(device, &alloc_info, nullptr, &memory));

  vkBindBufferMemory(device, buffer, memory, 0);
}

void VulkanEngine::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
{
  // Memory transfers are commands executed with buffers, just like drawing
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandPool = command_pool;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer command_buffer;
  vkAllocateCommandBuffers(device, &alloc_info, &command_buffer);

  // Start recording the command buffer
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(command_buffer, &begin_info);

  VkBufferCopy copy_region{};
  copy_region.srcOffset = 0;
  copy_region.dstOffset = 0;
  copy_region.size = size;
  vkCmdCopyBuffer(command_buffer, src, dst, 1, &copy_region);

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
