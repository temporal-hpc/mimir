#include "cudaview/engine/vk_cudadevice.hpp"

#include <cstring> // std::memcpy

#include "cudaview/vk_types.hpp"
#include "internal/vk_initializers.hpp"
#include "internal/validation.hpp"

CudaViewUnstructured VulkanCudaDevice::createUnstructuredBuffer(
  size_t elem_count, size_t elem_size, DataDomain domain, UnstructuredDataType type)
{
  CudaViewUnstructured mapped(elem_count, elem_size, domain, type);
  initBuffers(mapped.vertex_buffer, mapped.index_buffer);

  VkBufferUsageFlagBits usage;
  switch (mapped.data_type)
  {
    case UnstructuredDataType::Points: case UnstructuredDataType::Voxels:
    {
      usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; break;
    }
    case UnstructuredDataType::Edges:
    {
      usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT; break;
    }
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

CudaViewStructured VulkanCudaDevice::createStructuredBuffer(uint3 buffer_size,
  size_t elem_size, DataDomain domain, DataFormat format, StructuredDataType type)
{
  VkExtent3D extent{buffer_size.x, buffer_size.y, buffer_size.z};
  CudaViewStructured mapped(extent, elem_size, domain, format, type);
  initBuffers(mapped.vertex_buffer, mapped.index_buffer);

  if (mapped.data_type == StructuredDataType::Texture)
  {
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
    mapped.vk_sampler = createSampler(VK_FILTER_NEAREST, true);

    deletors.pushFunction([=]{
      validation::checkCuda(cudaDestroyExternalMemory(mapped.cuda_extmem));
      vkDestroyBuffer(logical_device, mapped.buffer.buffer, nullptr);
      vkFreeMemory(logical_device, mapped.buffer.memory, nullptr);
      vkDestroyImageView(logical_device, mapped.vk_view, nullptr);
      vkDestroyImage(logical_device, mapped.texture.image, nullptr);
      vkFreeMemory(logical_device, mapped.texture.memory, nullptr);
    });
  }
  else if (mapped.data_type == StructuredDataType::Voxels)
  {
    mapped.buffer = createExternalBuffer(mapped.element_size * mapped.element_count,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    );
    importCudaExternalMemory(&mapped.cuda_ptr, mapped.cuda_extmem,
      mapped.buffer.memory, mapped.element_size * mapped.element_count
    );

    auto buffer_size = sizeof(float3) * mapped.element_count;
    mapped.implicit = createBuffer(buffer_size,
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    float3 *data = nullptr;
    vkMapMemory(logical_device, mapped.implicit.memory, 0, buffer_size, 0, (void**)&data);
    auto slice_size = mapped.extent.width * mapped.extent.height;
    for (int z = 0; z < mapped.extent.depth; ++z)
    {
      auto rz = static_cast<float>(z) / mapped.extent.depth;
      for (int y = 0; y < mapped.extent.height; ++y)
      {
        auto ry = static_cast<float>(y) / mapped.extent.height;
        for (int x = 0; x < mapped.extent.width; ++x)
        {
          auto rx = static_cast<float>(x) / mapped.extent.width;
          data[slice_size * z + mapped.extent.width * y + x] = float3{rx, ry, rz};
        }
      }
    }
    vkUnmapMemory(logical_device, mapped.implicit.memory);

    deletors.pushFunction([=]{
      validation::checkCuda(cudaDestroyExternalMemory(mapped.cuda_extmem));
      vkDestroyBuffer(logical_device, mapped.buffer.buffer, nullptr);
      vkFreeMemory(logical_device, mapped.buffer.memory, nullptr);
      vkDestroyBuffer(logical_device, mapped.implicit.buffer, nullptr);
      vkFreeMemory(logical_device, mapped.implicit.memory, nullptr);
    });
  }

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

void VulkanCudaDevice::initBuffers(VulkanBuffer& vertex_buffer, VulkanBuffer& index_buffer)
{
  const std::vector<Vertex> vertices{
    { {  1.f,  1.f, 1.f }, { 1.f, 1.f } },
    { { -1.f,  1.f, 1.f }, { 0.f, 1.f } },
    { { -1.f, -1.f, 1.f }, { 0.f, 0.f } },
    { {  1.f, -1.f, 1.f }, { 1.f, 0.f } }/*,
    { {  1.f,  1.f, .5f }, { 1.f, 1.f } },
    { { -1.f,  1.f, .5f }, { 0.f, 1.f } },
    { { -1.f, -1.f, .5f }, { 0.f, 0.f } },
    { {  1.f, -1.f, .5f }, { 1.f, 0.f } }*/
  };
  // Indices for a single uv-mapped quad made from two triangles
  const std::vector<uint16_t> indices{ 0, 1, 2, 2, 3, 0, /*4, 5, 6, 6, 7, 4*/ };

  auto vert_size = sizeof(Vertex) * vertices.size();
  vertex_buffer = createBuffer(vert_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  );
  char *data = nullptr;
  vkMapMemory(logical_device, vertex_buffer.memory, 0, vert_size, 0, (void**)&data);
  std::memcpy(data, vertices.data(), vert_size);
  vkUnmapMemory(logical_device, vertex_buffer.memory);

  auto idx_size = sizeof(uint32_t) * indices.size();
  index_buffer = createBuffer(idx_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  );
  data = nullptr;
  vkMapMemory(logical_device, index_buffer.memory, 0, idx_size, 0, (void**)&data);
  std::memcpy(data, indices.data(), idx_size);
  vkUnmapMemory(logical_device, index_buffer.memory);

  deletors.pushFunction([=]{
    vkDestroyBuffer(logical_device, vertex_buffer.buffer, nullptr);
    vkFreeMemory(logical_device, vertex_buffer.memory, nullptr);
    vkDestroyBuffer(logical_device, index_buffer.buffer, nullptr);
    vkFreeMemory(logical_device, index_buffer.memory, nullptr);
  });
}
