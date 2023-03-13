#pragma once

#include "cudaview/engine/vk_device.hpp"

#include "cudaview/vk_types.hpp"
#include "cudaview/engine/cudaview.hpp"
#include "cudaview/engine/barrier.hpp"

// Class for encapsulating Vulkan device functions with Cuda interop 
// Inherited from VulkanDevice to encapsulate Cuda-related code
struct VulkanCudaDevice : public VulkanDevice
{
  // Use the constructor from the base VulkanDevice class
  using VulkanDevice::VulkanDevice;

  // Main library function, which setups all the visualization interop
  // TODO: Return the created view with a better handle
  CudaView createView(ViewParams params);
  // TODO: Currently called inside createView, but should be called from API
  InteropMemory getInteropBuffer(ViewParams params);
  InteropMemory getInteropImage(ViewParams params);

  void createUniformBuffers(CudaView& view, uint32_t img_count);
  void updateUniformBuffers(CudaView& view, uint32_t image_idx,
    ModelViewProjection mvp, PrimitiveParams options, SceneParams scene
  );
  InteropBarrier createInteropBarrier();
  void importCudaExternalMemory(cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size);
  void *getMemoryHandle(VkDeviceMemory memory,
    VkExternalMemoryHandleTypeFlagBits handle_type
  );
  void *getSemaphoreHandle(VkSemaphore semaphore,
    VkExternalSemaphoreHandleTypeFlagBits handle_type
  );
  void updateTexture(CudaView mapped);
  void loadTexture(CudaView& view, void *img_data);
};
