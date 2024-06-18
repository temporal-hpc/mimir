#pragma once

#include <vector> // std::vector

#include <mimir/engine/interop_view.hpp>

struct GLFWwindow;

namespace mimir
{

struct VulkanSwapchain
{
    VkSurfaceKHR surface     = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat color_format    = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR color_space;
    VkExtent2D extent;
    uint32_t image_count = 0;

    void create(uint32_t& width, uint32_t& height, PresentOptions opts,
        std::vector<uint32_t> queue_indices, VkPhysicalDevice gpu, VkDevice device
    );
    std::vector<VkImage> createImages(VkDevice device);
};

} // namespace mimir