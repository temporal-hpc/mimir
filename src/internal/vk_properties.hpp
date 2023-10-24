#pragma once

#include <vulkan/vulkan.h>

#include <iostream> // std::cout
#include <vector> // std::vector

namespace mimir
{
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
std::vector<const char*> getRequiredDeviceExtensions();

bool checkAllExtensionsSupported(VkPhysicalDevice dev,
    const std::vector<const char*>& device_extensions
);

bool isDeviceSuitable(VkPhysicalDevice dev, VkSurfaceKHR surface);

bool findQueueFamilies(VkPhysicalDevice dev, VkSurfaceKHR surface,
    uint32_t& graphics_family, uint32_t& present_family
);

} // namespace props
} // namespace mimir