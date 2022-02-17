#pragma once

#include <vulkan/vulkan.h>

struct VulkanTexture
{
  VkImage image         = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  uint32_t width        = 0;
  uint32_t height       = 0;
};
