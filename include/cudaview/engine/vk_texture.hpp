#pragma once

#include <vulkan/vulkan.h>

struct VulkanTexture
{
  VkImage image         = VK_NULL_HANDLE;
  VkImageLayout layout  = VK_IMAGE_LAYOUT_UNDEFINED;
  VkDeviceMemory memory = VK_NULL_HANDLE;

  uint32_t width  = 0;
  uint32_t height = 0;
  uint32_t depth  = 0;
};
