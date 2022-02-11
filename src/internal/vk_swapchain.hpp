#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>

struct VulkanSwapchain
{
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice logical_device;

  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkFormat color_format;
  VkColorSpaceKHR color_space;
  VkExtent2D swapchain_extent;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  uint32_t image_count = 0;
  std::vector<VkImage> images;
  std::vector<VkImageView> views;

  void initSurface(VkInstance instance, GLFWwindow *window);
  void connect(VkInstance instance, VkPhysicalDevice gpu, VkDevice device);
  void create(uint32_t& width, uint32_t& height, std::vector<uint32_t> queue_indices);
};
