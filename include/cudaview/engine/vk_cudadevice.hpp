#pragma once

#include "cudaview/engine/vk_device.hpp"

#include "cudaview/engine/cudaview.hpp"
#include "cudaview/engine/barrier.hpp"

struct VulkanCudaDevice : public VulkanDevice
{
  using VulkanDevice::VulkanDevice;

  CudaViewUnstructured createUnstructuredBuffer(size_t elem_count,
    size_t elem_size, DataDomain domain, UnstructuredDataType type
  );
  CudaViewStructured createStructuredBuffer(uint3 buffer_size, size_t elem_size,
    DataDomain domain, DataFormat format, StructuredDataType type
  );
  void initBuffers(VulkanBuffer& vertex_buffer, VulkanBuffer& index_buffer);
  void updateStructuredBuffer(CudaViewStructured mapped);
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
