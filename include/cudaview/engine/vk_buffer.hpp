#pragma once

#include <vulkan/vulkan.h>

struct VulkanBuffer
{
  VkBuffer buffer       = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size     = 0;
};
