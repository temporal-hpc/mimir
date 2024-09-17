#pragma once

#include <vulkan/vulkan.h>

#include <vector> // std::vector

namespace mimir
{

struct Swapchain
{
    VkSwapchainKHR current      = VK_NULL_HANDLE;
    VkSwapchainKHR old          = VK_NULL_HANDLE;
    VkFormat format             = VK_FORMAT_UNDEFINED;
    VkExtent2D extent           = { 0, 0 };
    uint32_t image_count        = 0;
    std::vector<VkImage> images = {};

    static Swapchain make(VkDevice device, VkPhysicalDevice ph_dev, VkSurfaceKHR surf,
        int width, int height, VkPresentModeKHR mode, std::vector<uint32_t> queue_indices
    );
};

static_assert(std::is_default_constructible_v<Swapchain>);
static_assert(std::is_nothrow_default_constructible_v<Swapchain>);

} // namespace mimir