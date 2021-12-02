#include "cudaview/vk_engine.hpp"
#include "validation.hpp"

#include <set> // std::set

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
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
  return extensions;
}

std::vector<const char*> VulkanEngine::getRequiredDeviceExtensions() const
{
  std::vector<const char*> extensions;
  extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
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

SwapchainSupportDetails VulkanEngine::getSwapchainProperties(
  VkPhysicalDevice dev) const
{
  SwapchainSupportDetails details;
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
