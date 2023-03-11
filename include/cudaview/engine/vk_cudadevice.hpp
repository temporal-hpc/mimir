#pragma once

#include "cudaview/engine/vk_device.hpp"

#include <vector>

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
  MappedMemory getMappedMemory(ViewParams params);

  // Get memory requirements for later allocation
  // TODO: Use span instead of vector
  VkMemoryRequirements getMemoryRequiements(VkBufferUsageFlags usage,
    const std::vector<uint32_t>& sizes
  );

  void createUniformBuffers(CudaView& view, uint32_t img_count);
  void updateUniformBuffers(CudaView& view, uint32_t image_idx,
    ModelViewProjection mvp, PrimitiveParams options, SceneParams scene
  );
  void updateTexture(CudaView mapped);
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
  void generateMipmaps(VkImage image, VkFormat img_format,
    int img_width, int img_height, int mip_levels
  );
};
