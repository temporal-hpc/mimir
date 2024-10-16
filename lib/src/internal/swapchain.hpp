#pragma once

#include <vulkan/vulkan.h>

#include <vector> // std::vector

namespace mimir
{

struct Swapchain
{
    // Swapchain handle currently in use
    VkSwapchainKHR current;
    // Handle to previously used swapchain object, made obsolete after a rebuild
    VkSwapchainKHR old;
    // Swapchain image format
    VkFormat format;
    // Swapchain image extent
    VkExtent2D extent;
    // Minimum amount of requested swapchain images
    uint32_t image_count;

    static Swapchain make(VkDevice device, VkPhysicalDevice ph_dev, VkSurfaceKHR surf,
        int width, int height, VkPresentModeKHR mode, std::vector<uint32_t> queue_indices
    );
    // Get swapchain image array with specified number of images
    std::vector<VkImage> getImages(VkDevice device);
};

static_assert(std::is_default_constructible_v<Swapchain>);
static_assert(std::is_nothrow_default_constructible_v<Swapchain>);
static_assert(std::is_trivially_default_constructible_v<Swapchain>);

} // namespace mimir