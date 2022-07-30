#pragma once

#include "cudaview/engine/vk_device.hpp"

#include "cudaview/engine/cudaview.hpp"
#include "cudaview/engine/barrier.hpp"

struct VulkanCudaDevice : public VulkanDevice
{
  using VulkanDevice::VulkanDevice;

  CudaView createUnstructuredView(ViewParams params);
  CudaView createStructuredView(ViewParams params);
  void initBuffers(VulkanBuffer& vertex_buffer, VulkanBuffer& index_buffer);
  void updateStructuredView(CudaView mapped);
  InteropBarrier createInteropBarrier();
  void importCudaExternalMemory(void **cuda_ptr,
    cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size
  );
  void *getMemoryHandle(VkDeviceMemory memory,
    VkExternalMemoryHandleTypeFlagBits handle_type
  );
  void *getSemaphoreHandle(VkSemaphore semaphore,
    VkExternalSemaphoreHandleTypeFlagBits handle_type
  );
};
