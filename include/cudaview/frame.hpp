#pragma once

#include <vulkan/vulkan.h>

struct FrameData
{
  VkSemaphore present_semaphore = VK_NULL_HANDLE;
  VkSemaphore render_semaphore  = VK_NULL_HANDLE;
  VkFence render_fence = VK_NULL_HANDLE;
};
