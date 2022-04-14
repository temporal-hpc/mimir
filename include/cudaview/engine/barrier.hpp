#pragma once

#include <vulkan/vulkan.h>

struct FrameBarrier
{
  VkSemaphore present_semaphore = VK_NULL_HANDLE;
  VkSemaphore render_semaphore  = VK_NULL_HANDLE;
  VkFence render_fence = VK_NULL_HANDLE;
};

struct InteropBarrier
{
  VkSemaphore vk_semaphore = VK_NULL_HANDLE;
  cudaExternalSemaphore_t cuda_semaphore = nullptr;
};
