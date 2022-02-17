#pragma once

#include "cudaview/engine/vk_device.hpp"

#include "cudaview/engine/vk_cudamem.hpp"

struct VulkanCudaDevice : public VulkanDevice
{
  using VulkanDevice::VulkanDevice;

  MappedUnstructuredMemory createUnstructuredBuffer(void **ptr_devmem,
    size_t elem_count, size_t elem_size, UnstructuredDataType type, DataDomain domain
  );
  MappedStructuredMemory createStructuredBuffer(void **ptr_devmem,
    size_t width, size_t height, size_t elem_size, DataFormat format
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
};
