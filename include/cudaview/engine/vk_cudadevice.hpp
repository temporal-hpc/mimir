#pragma once

#include "cudaview/engine/vk_device.hpp"

#include <memory> // std::shared_ptr

#include "cudaview/vk_types.hpp"
#include "cudaview/engine/barrier.hpp"

// Class for encapsulating Vulkan device functions with Cuda interop 
// Inherited from VulkanDevice to encapsulate Cuda-related code
struct VulkanCudaDevice : public VulkanDevice
{
  // Use the constructor from the base VulkanDevice class
  using VulkanDevice::VulkanDevice;

  InteropBarrier createInteropBarrier();
  void importCudaExternalMemory(cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size);
  void *getMemoryHandle(VkDeviceMemory memory,
    VkExternalMemoryHandleTypeFlagBits handle_type
  );
  void *getSemaphoreHandle(VkSemaphore semaphore,
    VkExternalSemaphoreHandleTypeFlagBits handle_type
  );
};
