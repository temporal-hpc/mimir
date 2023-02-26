#pragma once

#include "cudaview/engine/vk_device.hpp"
#include "cudaview/vk_types.hpp"

#include "cudaview/engine/cudaview.hpp"
#include "cudaview/engine/barrier.hpp"

struct VulkanCudaDevice : public VulkanDevice
{
  using VulkanDevice::VulkanDevice;

  CudaView createView(ViewParams params);
  void createUniformBuffers(CudaView& view, uint32_t img_count);
  void updateUniformBuffers(CudaView& view, uint32_t image_idx,
    glm::mat4 viewmat, glm::mat4 perspective
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
