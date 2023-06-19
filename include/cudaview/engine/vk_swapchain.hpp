#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector> // std::vector

#include <cudaview/deletion_queue.hpp>
#include <cudaview/engine/cudaview.hpp>

struct VulkanSwapchain
{
    VkSurfaceKHR surface     = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat color_format    = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR color_space;
    VkExtent2D extent;
    uint32_t image_count = 0;

    DeletionQueue main_deletors;
    DeletionQueue aux_deletors;

    ~VulkanSwapchain();
    void cleanup();
    void initSurface(VkInstance instance, GLFWwindow *window);
    void create(uint32_t& width, uint32_t& height, PresentOptions opts,
        std::vector<uint32_t> queue_indices, VkPhysicalDevice gpu, VkDevice device
    );
    std::vector<VkImage> createImages(VkDevice device);
};
