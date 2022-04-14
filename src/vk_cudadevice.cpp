#include "cudaview/engine/vk_cudadevice.hpp"

#include "internal/vk_initializers.hpp"
#include "internal/validation.hpp"

CudaViewUnstructured VulkanCudaDevice::createUnstructuredBuffer(
  size_t elem_count, size_t elem_size, DataDomain domain, UnstructuredDataType type)
{
  CudaViewUnstructured mapped(elem_count, elem_size, domain, type);

  VkBufferUsageFlagBits usage;
  if (mapped.data_type == UnstructuredDataType::Points)
  {
    usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  }
  else if (mapped.data_type == UnstructuredDataType::Edges)
  {
    usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  }
  // Init unstructured memory
  mapped.buffer = createExternalBuffer(mapped.element_size * mapped.element_count,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
  );
  importCudaExternalMemory(&mapped.cuda_ptr, mapped.cuda_extmem,
    mapped.buffer.memory, mapped.element_size * mapped.element_count
  );
  deletors.pushFunction([=]{
    validation::checkCuda(cudaDestroyExternalMemory(mapped.cuda_extmem));
    vkDestroyBuffer(logical_device, mapped.buffer.buffer, nullptr);
    vkFreeMemory(logical_device, mapped.buffer.memory, nullptr);
  });
  return mapped;
}

CudaViewStructured VulkanCudaDevice::createStructuredBuffer(
  uint3 buffer_size, size_t elem_size, DataDomain domain, DataFormat format)
{
  VkExtent3D extent{buffer_size.x, buffer_size.y, buffer_size.z};
  CudaViewStructured mapped(extent, elem_size, domain, format);

  // Init staging memory
  mapped.buffer = createExternalBuffer(mapped.element_size * mapped.element_count,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
  );
  importCudaExternalMemory(&mapped.cuda_ptr, mapped.cuda_extmem,
    mapped.buffer.memory, mapped.element_size * mapped.element_count
  );

  // Init texture memory
  auto img_type = getImageDimensions(domain);
  mapped.texture = createExternalImage(img_type, mapped.vk_format, extent,
    VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
  );
  transitionImageLayout(mapped.texture.image, mapped.vk_format,
    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
  );

  auto view_type = getViewType(domain);
  auto info = vkinit::imageViewCreateInfo(mapped.texture.image,
    view_type, mapped.vk_format, VK_IMAGE_ASPECT_COLOR_BIT
  );
  validation::checkVulkan(
    vkCreateImageView(logical_device, &info, nullptr, &mapped.vk_view)
  );

  mapped.vk_sampler = createSampler();

  deletors.pushFunction([=]{
    validation::checkCuda(cudaDestroyExternalMemory(mapped.cuda_extmem));
    vkDestroyBuffer(logical_device, mapped.buffer.buffer, nullptr);
    vkFreeMemory(logical_device, mapped.buffer.memory, nullptr);
    vkDestroyImageView(logical_device, mapped.vk_view, nullptr);
    vkDestroyImage(logical_device, mapped.texture.image, nullptr);
    vkFreeMemory(logical_device, mapped.texture.memory, nullptr);
  });

  return mapped;
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

void VulkanCudaDevice::importCudaExternalSemaphore(
  cudaExternalSemaphore_t& cuda_sem, VkSemaphore& vk_sem)
{
  cudaExternalSemaphoreHandleDesc sem_desc{};
  //sem_desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  sem_desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  sem_desc.handle.fd = (int)(uintptr_t)getSemaphoreHandle(
    vk_sem, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
  );
  sem_desc.flags = 0;
  validation::checkCuda(cudaImportExternalSemaphore(&cuda_sem, &sem_desc));
}

void VulkanCudaDevice::updateStructuredBuffer(CudaViewStructured mapped)
{
  auto src = mapped.buffer;
  auto dst = mapped.texture;
  transitionImageLayout(dst.image, mapped.vk_format,
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
  region.imageExtent = {dst.width, dst.height, dst.depth};
  immediateSubmit([=](VkCommandBuffer cmd)
  {
    vkCmdCopyBufferToImage(cmd, src.buffer, dst.image,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
    );
  });

  transitionImageLayout(dst.image, mapped.vk_format,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
  );
}
