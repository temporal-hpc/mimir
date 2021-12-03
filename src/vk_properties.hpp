#pragma once

#include <vulkan/vulkan.h>

#include <vector> // std::vector

namespace props
{

struct SwapchainSupportDetails
{
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

SwapchainSupportDetails getSwapchainProperties(
  VkPhysicalDevice dev, VkSurfaceKHR surface
);

// Handle additional extensions required by CUDA interop
std::vector<const char*> getRequiredExtensions();
std::vector<const char*> getRequiredDeviceExtensions();

bool isDeviceSuitable(VkPhysicalDevice dev, VkSurfaceKHR surface);

bool findQueueFamilies(VkPhysicalDevice dev, VkSurfaceKHR surface,
  uint32_t& graphics_family, uint32_t& present_family
);

bool checkAllExtensionsSupported(VkPhysicalDevice dev,
  const std::vector<const char*>& device_extensions
);

} // namespace props
