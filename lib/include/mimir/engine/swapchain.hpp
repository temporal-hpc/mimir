#pragma once

#include <vulkan/vulkan.h>

#include <vector> // std::vector

namespace mimir
{

struct Swapchain
{
    VkSwapchainKHR current;
    VkSwapchainKHR old;
    VkFormat format;
    VkExtent2D extent;
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