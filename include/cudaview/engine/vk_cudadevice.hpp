#pragma once

#include "cudaview/engine/vk_device.hpp"

#include "cudaview/engine/cudaview.hpp"

struct VulkanCudaDevice : public VulkanDevice
{
  using VulkanDevice::VulkanDevice;

  CudaViewUnstructured createUnstructuredBuffer(size_t elem_count,
    size_t elem_size, DataDomain domain, UnstructuredDataType type
  );
  CudaViewStructured createStructuredBuffer(uint3 buffer_size,
    size_t elem_size, DataDomain domain, DataFormat format
  );

  void importCudaExternalMemory(void **cuda_ptr,
    cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size
  );
  void *getMemoryHandle(VkDeviceMemory memory,
    VkExternalMemoryHandleTypeFlagBits handle_type
  );

  void importCudaExternalSemaphore(
    cudaExternalSemaphore_t& cuda_sem, VkSemaphore& vk_sem
  );
  void *getSemaphoreHandle(VkSemaphore semaphore,
    VkExternalSemaphoreHandleTypeFlagBits handle_type
  );
  void updateStructuredBuffer(CudaViewStructured mapped);
};
