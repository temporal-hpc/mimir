#include "cudaview/engine/vk_cudadevice.hpp"

#include <cuda_runtime.h>

#include <cstring> // std::memcpy

#include "cudaview/vk_types.hpp"
#include "internal/vk_initializers.hpp"
#include "internal/utils.hpp"
#include "internal/validation.hpp"

VkFormat getVulkanFormat(TextureFormat format)
{
  switch (format)
  {
    case TextureFormat::Uint8:   return VK_FORMAT_R8_UNORM;
    case TextureFormat::Int32:   return VK_FORMAT_R32_SINT;
    case TextureFormat::Float32: return VK_FORMAT_R32_SFLOAT;
    case TextureFormat::Rgba32:  return VK_FORMAT_R8G8B8A8_SRGB;
    default:                     return VK_FORMAT_UNDEFINED;
  }
}

VkImageType getImageType(DataDomain domain)
{
  switch (domain)
  {
    case DataDomain::Domain2D: return VK_IMAGE_TYPE_2D;
    case DataDomain::Domain3D: return VK_IMAGE_TYPE_3D;
    default:                   return VK_IMAGE_TYPE_1D;
  }
}

VkImageViewType getViewType(DataDomain domain)
{
  switch (domain)
  {
    case DataDomain::Domain2D: return VK_IMAGE_VIEW_TYPE_2D;
    case DataDomain::Domain3D: return VK_IMAGE_VIEW_TYPE_3D;
    default:                   return VK_IMAGE_VIEW_TYPE_1D;
  }
}

VkBufferUsageFlags getUsageFlags(PrimitiveType p, ResourceType r)
{
  if (r == ResourceType::TextureLinear) return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  switch (p)
  {
    case PrimitiveType::Points: case PrimitiveType::Voxels:
    {
      return usage | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    }
    case PrimitiveType::Edges:
    {
      return usage | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    }
    default: return usage;
  }
}

CudaView VulkanCudaDevice::createView(ViewParams params)
{
  CudaView view;
  view.params = params;
  view.vk_format = getVulkanFormat(params.texture_format);
  view.vk_extent = {params.extent.x, params.extent.y, params.extent.z};
  VkDeviceSize memsize = params.element_size * params.element_count;

  if (params.resource_type == ResourceType::Texture)
  {
    size_t image_width = params.extent.x;
    size_t image_height = params.extent.y;

    // Init texture memory
    auto img_type = getImageType(params.data_domain);
    VkExternalMemoryImageCreateInfo ext_info{};
    ext_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    ext_info.pNext = nullptr;
    ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    view.image = createImage(img_type, view.vk_format, view.vk_extent,
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      &ext_info
    );

    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(logical_device, view.image, &mem_req);

    VkExportMemoryAllocateInfoKHR export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    export_info.pNext = nullptr;
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = &export_info;
    alloc_info.allocationSize = mem_req.size;
    auto mem_props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    alloc_info.memoryTypeIndex = findMemoryType(mem_req.memoryTypeBits, mem_props);

    validation::checkVulkan(
      vkAllocateMemory(logical_device, &alloc_info, nullptr, &view.img_memory)
    );
    vkBindImageMemory(logical_device, view.image, view.img_memory, 0);

    cudaExternalMemoryHandleDesc extmem_desc{};
    extmem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    extmem_desc.size = mem_req.size;
    extmem_desc.handle.fd = (int)(uintptr_t)getMemoryHandle(
      view.img_memory, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    );

    validation::checkCuda(cudaImportExternalMemory(&view.cuda_extmem, &extmem_desc));

    auto cuda_extent = make_cudaExtent(image_width, image_height, 0);
    cudaChannelFormatDesc format_desc;
    format_desc.x = 8;
    format_desc.y = 8;
    format_desc.z = 8;
    format_desc.w = 8;
    format_desc.f = cudaChannelFormatKindUnsigned;

    cudaExternalMemoryMipmappedArrayDesc array_desc{};
    array_desc.offset     = 0;
    array_desc.formatDesc = format_desc;
    array_desc.extent     = cuda_extent;
    array_desc.flags      = 0;
    array_desc.numLevels  = 1; // mipLevels

    validation::checkCuda(cudaExternalMemoryGetMappedMipmappedArray(
      &view.cudaMipmappedImageArray, view.cuda_extmem, &array_desc)
    );

    validation::checkCuda(cudaMallocMipmappedArray(
      &view.cudaMipmappedImageArrayTemp, &format_desc, cuda_extent, 1 // mipLevels
    ));
    validation::checkCuda(cudaMallocMipmappedArray(
      &view.cudaMipmappedImageArrayOrig, &format_desc, cuda_extent, 1 // mipLevels
    ));

    for (int level_idx; level_idx < 1; ++level_idx)
    {
      cudaArray_t mipLevelArray, mipLevelArrayTemp, mipLevelArrayOrig;

      validation::checkCuda(cudaGetMipmappedArrayLevel(
        &mipLevelArray, view.cudaMipmappedImageArray, level_idx
      ));
      validation::checkCuda(cudaGetMipmappedArrayLevel(
        &mipLevelArrayTemp, view.cudaMipmappedImageArrayTemp, level_idx
      ));
      validation::checkCuda(cudaGetMipmappedArrayLevel(
        &mipLevelArrayOrig, view.cudaMipmappedImageArrayOrig, level_idx
      ));

      uint32_t width = (image_width >> level_idx) ? (image_width >> level_idx) : 1;
      uint32_t height = (image_height >> level_idx) ? (image_height >> level_idx) : 1;
      validation::checkCuda(cudaMemcpy2DArrayToArray(
        mipLevelArrayOrig, 0, 0, mipLevelArray, 0, 0,
        width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice
      ));

      cudaResourceDesc res_desc{};
      res_desc.resType = cudaResourceTypeArray;
      res_desc.res.array.array = mipLevelArrayTemp;

      cudaSurfaceObject_t surf_temp{};
      validation::checkCuda(cudaCreateSurfaceObject(&surf_temp, &res_desc));
      view.surfaceObjectListTemp.push_back(surf_temp);
    }

    cudaResourceDesc res_desc{};
    res_desc.resType = cudaResourceTypeMipmappedArray;
    res_desc.res.mipmap.mipmap = view.cudaMipmappedImageArrayOrig;

    cudaTextureDesc tex_desc{};
    tex_desc.normalizedCoords = true;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.maxMipmapLevelClamp = 0.f; // float(mipLevels - 1)
    tex_desc.readMode = cudaReadModeNormalizedFloat;

    validation::checkCuda(cudaCreateTextureObject(
      &view.textureObjMipMapInput, &res_desc, &tex_desc, nullptr
    ));

    validation::checkCuda(cudaMalloc(
      &view.d_surfaceObjectList, sizeof(cudaSurfaceObject_t) // * mipLevels
    ));
    validation::checkCuda(cudaMalloc(
      &view.d_surfaceObjectListTemp, sizeof(cudaSurfaceObject_t) // * mipLevels
    ));
    printf("%lu %lu\n", view.surfaceObjectList.size(), view.surfaceObjectListTemp.size());
    validation::checkCuda(cudaMemcpy(
      &view.d_surfaceObjectList, view.surfaceObjectList.data(),
      sizeof(cudaSurfaceObject_t), cudaMemcpyHostToDevice
    ));
    validation::checkCuda(cudaMemcpy(
      &view.d_surfaceObjectListTemp, view.surfaceObjectListTemp.data(),
      sizeof(cudaSurfaceObject_t), cudaMemcpyHostToDevice
    ));

    deletors.pushFunction([=]{
      validation::checkCuda(cudaFreeMipmappedArray(view.cudaMipmappedImageArrayTemp));
      validation::checkCuda(cudaFreeMipmappedArray(view.cudaMipmappedImageArrayOrig));
      validation::checkCuda(cudaFreeMipmappedArray(view.cudaMipmappedImageArray));
      validation::checkCuda(cudaDestroyTextureObject(view.textureObjMipMapInput));
    });
    return view;
  }

  auto usage = getUsageFlags(params.primitive_type, params.resource_type);

  VkExternalMemoryBufferCreateInfo extmem_info{};
  extmem_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  extmem_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  view.interop_buffer = createBuffer(memsize, usage, &extmem_info);

  VkExportMemoryAllocateInfoKHR export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
  export_info.pNext = nullptr;
  export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  VkMemoryRequirements requirements;
  vkGetBufferMemoryRequirements(logical_device, view.interop_buffer, &requirements);
  view.interop_memory = allocateMemory(requirements,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &export_info
  );

  vkBindBufferMemory(logical_device, view.interop_buffer, view.interop_memory, 0);
  importCudaExternalMemory(
    &view.cuda_ptr, view.cuda_extmem, view.interop_memory, memsize
  );
  deletors.pushFunction([=]{
    validation::checkCuda(cudaDestroyExternalMemory(view.cuda_extmem));
    vkDestroyBuffer(logical_device, view.interop_buffer, nullptr);
    vkFreeMemory(logical_device, view.interop_memory, nullptr);
  });

  if (params.resource_type == ResourceType::TextureLinear)
  {
    const std::vector<Vertex> vertices{
      { {  1.f,  1.f, 0.f }, { 1.f, 1.f } },
      { { -1.f,  1.f, 0.f }, { 0.f, 1.f } },
      { { -1.f, -1.f, 0.f }, { 0.f, 0.f } },
      { {  1.f, -1.f, 0.f }, { 1.f, 0.f } }/*,
      { {  1.f,  1.f, .5f }, { 1.f, 1.f } },
      { { -1.f,  1.f, .5f }, { 0.f, 1.f } },
      { { -1.f, -1.f, .5f }, { 0.f, 0.f } },
      { {  1.f, -1.f, .5f }, { 1.f, 0.f } }*/
    };
    // Indices for a single uv-view quad made from two triangles
    const std::vector<uint16_t> indices{ 0, 1, 2, 2, 3, 0};//, 4, 5, 6, 6, 7, 4 };

    auto vert_size = sizeof(Vertex) * vertices.size();
    auto ids_size = sizeof(uint16_t) * indices.size();

    // Test buffer for asking about its memory properties
    usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    auto test_buffer = createBuffer(1, usage);
    vkGetBufferMemoryRequirements(logical_device, test_buffer, &requirements);
    auto vert_size_align = getAlignedSize(vert_size, requirements.alignment);
    auto ids_size_align = getAlignedSize(ids_size, requirements.alignment);
    requirements.size = vert_size_align + ids_size_align;
    vkDestroyBuffer(logical_device, test_buffer, nullptr);

    // Allocate memory and bind it to buffers
    view.aux_memory = allocateMemory(requirements,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    view.vertex_buffer = createBuffer(vert_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    vkBindBufferMemory(logical_device, view.vertex_buffer, view.aux_memory, 0);
    view.index_buffer = createBuffer(ids_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    vkBindBufferMemory(logical_device, view.index_buffer, view.aux_memory, vert_size_align);

    char *data = nullptr;
    vkMapMemory(logical_device, view.aux_memory, 0, vert_size, 0, (void**)&data);
    std::memcpy(data, vertices.data(), vert_size);
    vkUnmapMemory(logical_device, view.aux_memory);

    data = nullptr;
    vkMapMemory(logical_device, view.aux_memory, vert_size_align, ids_size, 0, (void**)&data);
    std::memcpy(data, indices.data(), ids_size);
    vkUnmapMemory(logical_device, view.aux_memory);

    deletors.pushFunction([=]{
      vkDestroyBuffer(logical_device, view.vertex_buffer, nullptr);
      vkDestroyBuffer(logical_device, view.index_buffer, nullptr);
      vkFreeMemory(logical_device, view.aux_memory, nullptr);
    });

    // Init texture memory
    auto img_type = getImageType(params.data_domain);
    VkExternalMemoryImageCreateInfo ext_info{};
    ext_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    ext_info.pNext = nullptr;
    ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    view.image = createImage(img_type, view.vk_format, view.vk_extent,
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      &ext_info
    );

    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(logical_device, view.image, &mem_req);

    VkExportMemoryAllocateInfoKHR export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    export_info.pNext = nullptr;
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = &export_info;
    alloc_info.allocationSize = mem_req.size;
    auto mem_props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    alloc_info.memoryTypeIndex = findMemoryType(mem_req.memoryTypeBits, mem_props);

    validation::checkVulkan(
      vkAllocateMemory(logical_device, &alloc_info, nullptr, &view.img_memory)
    );
    vkBindImageMemory(logical_device, view.image, view.img_memory, 0);

    transitionImageLayout(view.image, view.vk_format,
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    auto view_type = getViewType(params.data_domain);
    auto info = vkinit::imageViewCreateInfo(view.image,
      view_type, view.vk_format, VK_IMAGE_ASPECT_COLOR_BIT
    );
    validation::checkVulkan(
      vkCreateImageView(logical_device, &info, nullptr, &view.vk_view)
    );
    view.vk_sampler = createSampler(VK_FILTER_NEAREST, true);

    deletors.pushFunction([=]{
      vkDestroyImageView(logical_device, view.vk_view, nullptr);
      vkDestroyImage(logical_device, view.image, nullptr);
      vkFreeMemory(logical_device, view.img_memory, nullptr);
    });
  }
  else if (params.resource_type == ResourceType::StructuredBuffer)
  {
    auto buffer_size = sizeof(float3) * params.element_count;

    // Test buffer for asking about its memory properties
    auto test_buffer = createBuffer(1, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    vkGetBufferMemoryRequirements(logical_device, test_buffer, &requirements);
    auto vert_size_align = getAlignedSize(buffer_size, requirements.alignment);
    requirements.size = vert_size_align;
    vkDestroyBuffer(logical_device, test_buffer, nullptr);

    // Allocate memory and bind it to buffers
    view.aux_memory = allocateMemory(requirements,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    view.vertex_buffer = createBuffer(buffer_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    vkBindBufferMemory(logical_device, view.vertex_buffer, view.aux_memory, 0);

    float3 *data = nullptr;
    vkMapMemory(logical_device, view.aux_memory, 0, buffer_size, 0, (void**)&data);
    auto slice_size = view.vk_extent.width * view.vk_extent.height;
    for (uint32_t z = 0; z < view.vk_extent.depth; ++z)
    {
      auto rz = static_cast<float>(z) / view.vk_extent.depth;
      for (uint32_t y = 0; y < view.vk_extent.height; ++y)
      {
        auto ry = static_cast<float>(y) / view.vk_extent.height;
        for (uint32_t x = 0; x < view.vk_extent.width; ++x)
        {
          auto rx = static_cast<float>(x) / view.vk_extent.width;
          data[slice_size * z + view.vk_extent.width * y + x] = float3{rx, ry, rz};
        }
      }
    }
    vkUnmapMemory(logical_device, view.aux_memory);

    deletors.pushFunction([=]{
      vkDestroyBuffer(logical_device, view.vertex_buffer, nullptr);
      vkFreeMemory(logical_device, view.aux_memory, nullptr);
    });
  }
  return view;
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

void VulkanCudaDevice::updateStructuredView(CudaView view)
{
  transitionImageLayout(view.image, view.vk_format,
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
    vkCmdCopyBufferToImage(cmd, view.interop_buffer, view.image,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
    );
  });

  transitionImageLayout(view.image, view.vk_format,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
  );
}
