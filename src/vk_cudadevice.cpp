#include "cudaview/engine/vk_cudadevice.hpp"

#include <cuda_runtime.h>

#include <cstring> // std::memcpy

#include "cudaview/vk_types.hpp"
#include "internal/vk_initializers.hpp"
#include "internal/color.hpp"
#include "internal/utils.hpp"
#include "internal/validation.hpp"

void VulkanCudaDevice::generateMipmaps(VkImage image, VkFormat img_format,
  int img_width, int img_height, int mip_levels)
{
  VkFormatProperties format_props;
  vkGetPhysicalDeviceFormatProperties(physical_device, img_format, &format_props);

  if (!(format_props.optimalTilingFeatures &
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
  {
    throw std::runtime_error(
      "texture image format does not support linear blitting!");
  }

  immediateSubmit([=](VkCommandBuffer cmd)
  {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mip_width  = img_width;
    int32_t mip_height = img_height;

    for (int i = 1; i < mip_levels; i++)
    {
      barrier.subresourceRange.baseMipLevel = i - 1;
      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                           nullptr, 1, &barrier);

      VkImageBlit blit = {};
      blit.srcOffsets[0] = {0, 0, 0};
      blit.srcOffsets[1] = {mip_width, mip_height, 1};
      blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.srcSubresource.mipLevel = i - 1;
      blit.srcSubresource.baseArrayLayer = 0;
      blit.srcSubresource.layerCount = 1;
      blit.dstOffsets[0] = {0, 0, 0};
      blit.dstOffsets[1] = {mip_width > 1 ? mip_width / 2 : 1,
                            mip_height > 1 ? mip_height / 2 : 1, 1};
      blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.dstSubresource.mipLevel = i;
      blit.dstSubresource.baseArrayLayer = 0;
      blit.dstSubresource.layerCount = 1;

      vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                     VK_FILTER_LINEAR);

      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                           0, nullptr, 1, &barrier);

      if (mip_width > 1) mip_width /= 2;
      if (mip_height > 1) mip_height /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mip_levels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
      0, nullptr, 1, &barrier);
  });
}

void VulkanCudaDevice::createUniformBuffers(CudaView& view, uint32_t img_count)
{
  auto min_alignment = properties.limits.minUniformBufferOffsetAlignment;
  auto size_mvp = getAlignedSize(sizeof(ModelViewProjection), min_alignment);
  auto size_options = getAlignedSize(sizeof(PrimitiveParams), min_alignment);
  auto size_scene = getAlignedSize(sizeof(SceneParams), min_alignment);

  VkDeviceSize buffer_size = img_count * (2 * size_mvp + size_options + size_scene);

  auto test_buffer = createBuffer(1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(logical_device, test_buffer, &requirements);
  requirements.size = buffer_size;
  vkDestroyBuffer(logical_device, test_buffer, nullptr);

  // Allocate memory and bind it to buffers
  view.ubo_memory = allocateMemory(requirements,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  );
  view.ubo_buffer = createBuffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  vkBindBufferMemory(logical_device, view.ubo_buffer, view.ubo_memory, 0);
}

void VulkanCudaDevice::updateUniformBuffers(CudaView& view, uint32_t image_idx,
  ModelViewProjection mvp, PrimitiveParams options, SceneParams scene)
{
  auto min_alignment = properties.limits.minUniformBufferOffsetAlignment;
  auto size_mvp = getAlignedSize(sizeof(ModelViewProjection), min_alignment);
  auto size_options = getAlignedSize(sizeof(PrimitiveParams), min_alignment);
  auto size_scene = getAlignedSize(sizeof(SceneParams), min_alignment);
  auto size_ubo = 2 * size_mvp + size_options + size_scene;
  auto offset = image_idx * size_ubo;

  char *data = nullptr;
  vkMapMemory(logical_device, view.ubo_memory, offset, size_ubo, 0, (void**)&data);
  std::memcpy(data, &mvp, sizeof(mvp));
  std::memcpy(data + size_mvp, &options, sizeof(options));
  std::memcpy(data + size_mvp + size_options, &scene, sizeof(scene));
  std::memcpy(data + size_mvp + size_options + size_scene, &mvp, sizeof(mvp));
  vkUnmapMemory(logical_device, view.ubo_memory);
}

void *VulkanCudaDevice::getMemoryHandle(VkDeviceMemory memory,
  VkExternalMemoryHandleTypeFlagBits handle_type)
{
  int fd = -1;

  VkMemoryGetFdInfoKHR fd_info{};
  fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  fd_info.pNext = nullptr;
  fd_info.memory = memory;
  fd_info.handleType = handle_type;

  auto fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(
    logical_device, "vkGetMemoryFdKHR"
  );
  if (!fpGetMemoryFdKHR)
  {
    throw std::runtime_error("Failed to retrieve function!");
  }
  if (fpGetMemoryFdKHR(logical_device, &fd_info, &fd) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }
  return (void*)(uintptr_t)fd;
}

void VulkanCudaDevice::importCudaExternalMemory(void **cuda_ptr,
  cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size)
{
  cudaExternalMemoryHandleDesc extmem_desc{};
  extmem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  extmem_desc.size = size;
  extmem_desc.handle.fd = (int)(uintptr_t)getMemoryHandle(
    vk_mem, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
  );

  validation::checkCuda(cudaImportExternalMemory(&cuda_mem, &extmem_desc));

  cudaExternalMemoryBufferDesc buffer_desc{};
  buffer_desc.offset = 0;
  buffer_desc.size   = size;
  buffer_desc.flags  = 0;

  validation::checkCuda(cudaExternalMemoryGetMappedBuffer(
    cuda_ptr, cuda_mem, &buffer_desc)
  );
}

void *VulkanCudaDevice::getSemaphoreHandle(VkSemaphore semaphore,
  VkExternalSemaphoreHandleTypeFlagBits handle_type)
{
  int fd;
  VkSemaphoreGetFdInfoKHR fd_info{};
  fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  fd_info.pNext = nullptr;
  fd_info.semaphore  = semaphore;
  fd_info.handleType = handle_type;

  auto fpGetSemaphore = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
    logical_device, "vkGetSemaphoreFdKHR"
  );
  if (!fpGetSemaphore)
  {
    throw std::runtime_error("Failed to retrieve semaphore function handle!");
  }
  validation::checkVulkan(fpGetSemaphore(logical_device, &fd_info, &fd));

  return (void*)(uintptr_t)fd;
}

InteropBarrier VulkanCudaDevice::createInteropBarrier()
{
  /*VkSemaphoreTypeCreateInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_info.pNext = nullptr;
  timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_info.initialValue = 0;*/

  VkExportSemaphoreCreateInfoKHR export_info{};
  export_info.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
  export_info.pNext       = nullptr; // &timeline_info
  export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

  auto semaphore_info  = vkinit::semaphoreCreateInfo();
  semaphore_info.pNext = &export_info;

  InteropBarrier barrier;
  validation::checkVulkan(vkCreateSemaphore(
    logical_device, &semaphore_info, nullptr, &barrier.vk_semaphore)
  );

  cudaExternalSemaphoreHandleDesc desc{};
  //desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  desc.handle.fd = (int)(uintptr_t)getSemaphoreHandle(
    barrier.vk_semaphore, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
  );
  desc.flags = 0;
  validation::checkCuda(cudaImportExternalSemaphore(&barrier.cuda_semaphore, &desc));

  deletors.pushFunction([=]{
    validation::checkCuda(cudaDestroyExternalSemaphore(barrier.cuda_semaphore));
    vkDestroySemaphore(logical_device, barrier.vk_semaphore, nullptr);
  });
  return barrier;
}

void VulkanCudaDevice::updateTexture(CudaView view)
{
  transitionImageLayout(view.image,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
  );

  VkImageSubresourceLayers subres;
  subres.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  subres.mipLevel       = 0;
  subres.baseArrayLayer = 0;
  subres.layerCount     = 1;

  VkBufferImageCopy region;
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource = subres;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = view.vk_extent;
  immediateSubmit([=](VkCommandBuffer cmd)
  {
    vkCmdCopyBufferToImage(cmd, view._interop.data_buffer, view.image,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
    );
  });

  transitionImageLayout(view.image,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
  );
}
