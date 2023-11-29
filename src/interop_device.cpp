#include <mimir/engine/interop_device.hpp>

#include <cuda_runtime.h> // make_cudaExtent
#include <cstring> // std::memcpy

#include <mimir/shader_types.hpp>
#include <mimir/validation.hpp>
#include "internal/vk_initializers.hpp"

namespace mimir
{

VkImageTiling getImageTiling(ResourceType type)
{
    switch (type)
    {
        case ResourceType::Texture: return VK_IMAGE_TILING_OPTIMAL;
        default:                    return VK_IMAGE_TILING_LINEAR;
    }
}

// Converts a InteropView image type to its Vulkan equivalent
VkImageType getImageType(DataLayout layout)
{
    switch (layout)
    {
        case DataLayout::Layout2D: return VK_IMAGE_TYPE_2D;
        case DataLayout::Layout3D: return VK_IMAGE_TYPE_3D;
        default:                   return VK_IMAGE_TYPE_1D;
    }
}

// Converts a InteropView layout type to its Vulkan equivalent
VkImageViewType getImageViewType(DataLayout layout)
{
    switch (layout)
    {
        case DataLayout::Layout2D: return VK_IMAGE_VIEW_TYPE_2D;
        case DataLayout::Layout3D: return VK_IMAGE_VIEW_TYPE_3D;
        default:                   return VK_IMAGE_VIEW_TYPE_1D;
    }
}

VkBufferUsageFlags getBufferUsage(ResourceType type)
{
    // Every interop buffer should at least use the DST_BIT flags,
    // to allow for memcpy operations
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    switch (type)
    {
        case ResourceType::Buffer: return usage | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        case ResourceType::IndexBuffer: return usage | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        default: return usage;
    }
}

void initImplicitCoords(VkDevice dev, VkDeviceMemory mem, VkDeviceSize memsize, uint3 extent)
{
    float3 *data = nullptr;
    vkMapMemory(dev, mem, 0, memsize, 0, (void**)&data);
    auto slice_size = extent.x * extent.y;
    for (uint32_t z = 0; z < extent.z; ++z)
    {
        auto rz = static_cast<float>(z) / extent.z;
        for (uint32_t y = 0; y < extent.y; ++y)
        {
            auto ry = static_cast<float>(y) / extent.y;
            for (uint32_t x = 0; x < extent.x; ++x)
            {
                auto rx = static_cast<float>(x) / extent.x;
                data[slice_size * z + extent.x * y + x] = float3{rx, ry, rz};
            }
        }
    }
    vkUnmapMemory(dev, mem);
}

cudaMipmappedArray_t createMipmapArray(cudaExternalMemory_t cuda_extmem, MemoryParams params)
{
    constexpr int level_count = 1; // TODO: Should be a parameter

    cudaChannelFormatDesc format_desc;
    format_desc.x = 8;
    format_desc.y = 8;
    format_desc.z = 8;
    format_desc.w = 8;
    format_desc.f = cudaChannelFormatKindUnsigned;

    auto sz = getSize(params.element_count, params.layout);
    size_t image_width  = sz.x;
    size_t image_height = sz.y;
    auto cuda_extent = make_cudaExtent(image_width, image_height, 0);

    cudaExternalMemoryMipmappedArrayDesc array_desc{};
    array_desc.offset     = 0;
    array_desc.formatDesc = format_desc;
    array_desc.extent     = cuda_extent;
    array_desc.flags      = 0;
    array_desc.numLevels  = level_count;

    cudaMipmappedArray_t mipmap_array;
    validation::checkCuda(cudaExternalMemoryGetMappedMipmappedArray(
        &mipmap_array, cuda_extmem, &array_desc)
    );
    return mipmap_array;
}

VkImage InteropDevice::createImage(MemoryParams params)
{
    auto img_type = getImageType(params.layout);
    auto format   = getDataFormat(params.data_type, params.channel_count);
    auto usage    = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    auto tiling   = VK_IMAGE_TILING_OPTIMAL;
    auto sz = getSize(params.element_count, params.layout);
    VkExtent3D extent = {sz.x, sz.y, sz.z};

    VkExternalMemoryImageCreateInfo extmem_info{};
    extmem_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    extmem_info.pNext = nullptr;
    extmem_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    // TODO: Check if texture is within bounds
    //auto max_img_dim = properties.limits.maxImageDimension3D;

    // Check that the upcoming image parameters are supported
    VkPhysicalDeviceImageFormatInfo2 format_info{};
    format_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    format_info.pNext = nullptr;
    format_info.format = format;
    format_info.type = img_type;
    format_info.tiling = tiling;
    format_info.usage = usage;
    format_info.flags = 0;
    VkImageFormatProperties2 format_props{};
    format_props.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
    // TODO: Do not rely on validation layers to stop invalid image formats
    validation::checkVulkan(vkGetPhysicalDeviceImageFormatProperties2(
        physical_device, &format_info, &format_props
    ));

    auto info = vkinit::imageCreateInfo(img_type, format, extent, usage);
    info.pNext         = &extmem_info;
    info.flags         = 0;
    info.tiling        = tiling;
    info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateImage(logical_device, &info, nullptr, &image));
    return image;
}

void InteropDevice::initMemoryBuffer(InteropMemory& interop)
{
    const auto &params = interop.params;
    const auto element_size = getDataSize(params.data_type, params.channel_count);
    const auto element_count = getElementCount(params.element_count, params.layout);

    // Create external memory buffers
    VkDeviceSize memsize = element_size * element_count;
    auto usage = getBufferUsage(params.resource_type);
    VkExternalMemoryBufferCreateInfo extmem_info{};
    extmem_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extmem_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    interop.data_buffer = createBuffer(memsize, usage, &extmem_info);
    deletors.add([=,this]{
        vkDestroyBuffer(logical_device, interop.data_buffer, nullptr);
    });
    VkMemoryRequirements memreq{};
    vkGetBufferMemoryRequirements(logical_device, interop.data_buffer, &memreq);

    // Create and export (to CUDA) the memory allocated with the
    // requirements obtained above
    VkExportMemoryAllocateInfoKHR export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    export_info.pNext = nullptr;
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    auto memflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    interop.memory = allocateMemory(memreq, memflags, &export_info);
    interop.cuda_extmem = importCudaExternalMemory(interop.memory, memreq.size);
    deletors.add([=,this]{
        validation::checkCuda(cudaDestroyExternalMemory(interop.cuda_extmem));
        vkFreeMemory(logical_device, interop.memory, nullptr);
    });

    // Bind the resources to the external memory allocated above
    vkBindBufferMemory(logical_device, interop.data_buffer, interop.memory, 0);
    cudaExternalMemoryBufferDesc buffer_desc{};
    buffer_desc.offset = 0;
    buffer_desc.size   = memsize;
    buffer_desc.flags  = 0;
    validation::checkCuda(cudaExternalMemoryGetMappedBuffer(
        &interop.cuda_ptr, interop.cuda_extmem, &buffer_desc)
    );
}

void InteropDevice::initViewBuffer(InteropView& view)
{
    const auto &params = view.params;
    // For structured domain views, initialize auxiliary resources and memory
    if (params.domain_type == DomainType::Structured)
    {
        // Allocate memory and bind it to buffers
        auto buffer_size = sizeof(float3) * params.element_count;
        view.aux_buffer = createBuffer(buffer_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        VkMemoryRequirements memreq{};
        vkGetBufferMemoryRequirements(logical_device, view.aux_buffer, &memreq);
        auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        view.aux_memory = allocateMemory(memreq, flags);
        deletors.add([=,this]{
            vkDestroyBuffer(logical_device, view.aux_buffer, nullptr);
            vkFreeMemory(logical_device, view.aux_memory, nullptr);
        });
        vkBindBufferMemory(logical_device, view.aux_buffer, view.aux_memory, 0);

        initImplicitCoords(logical_device, view.aux_memory, buffer_size, params.extent);
        view.vert_buffers.push_back(view.aux_buffer);
        view.buffer_offsets.push_back(0);
    }

    for (const auto &[attr, memory] : view.params.attributes)
    {
        if (attr == AttributeType::Index)
        {
            view.idx_buffer = memory.data_buffer;
            view.idx_type = getIndexType(memory.params.data_type);
        }
        else
        {
            view.vert_buffers.push_back(memory.data_buffer);
            view.buffer_offsets.push_back(0);
        }
    }
}

void InteropDevice::initMemoryImage(InteropMemory& interop)
{
    const auto &params = interop.params;
    auto sz = getSize(params.element_count, params.layout);
    interop.vk_extent = {sz.x, sz.y, sz.z};
    interop.vk_format = getDataFormat(params.data_type, params.channel_count);

    // Init texture memory
    interop.image = createImage(params);
    interop.vk_sampler = createSampler(VK_FILTER_NEAREST, true);
    deletors.add([=,this]{
        vkDestroyImage(logical_device, interop.image, nullptr);
    });
    VkMemoryRequirements memreq{};
    vkGetImageMemoryRequirements(logical_device, interop.image, &memreq);

    auto memflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkExportMemoryAllocateInfoKHR export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    export_info.pNext = nullptr;
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    interop.image_memory = allocateMemory(memreq, memflags, &export_info);
    interop.cuda_extmem = importCudaExternalMemory(interop.image_memory, memreq.size);
    deletors.add([=,this]{
        validation::checkCuda(cudaDestroyExternalMemory(interop.cuda_extmem));
        vkFreeMemory(logical_device, interop.image_memory, nullptr);
    });

    vkBindImageMemory(logical_device, interop.image, interop.image_memory, 0);

    auto view_type = getImageViewType(params.layout);
    auto info = vkinit::imageViewCreateInfo(interop.image,
        view_type, interop.vk_format, VK_IMAGE_ASPECT_COLOR_BIT
    );
    validation::checkVulkan(vkCreateImageView(logical_device, &info, nullptr, &interop.vk_view));
    deletors.add([=,this]{
        vkDestroyImageView(logical_device, interop.vk_view, nullptr);
    });

    transitionImageLayout(interop.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );
    interop.mipmap_array = createMipmapArray(interop.cuda_extmem, params);
    deletors.add([=,this]{
        validation::checkCuda(cudaFreeMipmappedArray(interop.mipmap_array));
    });
}

void InteropDevice::initMemoryImageLinear(InteropMemory& interop)
{
    initMemoryBuffer(interop);

    const auto &params = interop.params;
    auto sz = getSize(params.element_count, params.layout);
    interop.vk_extent = {sz.x, sz.y, sz.z};
    interop.vk_format = getDataFormat(params.data_type, params.channel_count);

    // Init texture memory
    interop.image = createImage(params);
    interop.vk_sampler = createSampler(VK_FILTER_NEAREST, true);
    deletors.add([=,this]{
        vkDestroyImage(logical_device, interop.image, nullptr);
    });

    VkMemoryRequirements memreq{};
    vkGetImageMemoryRequirements(logical_device, interop.image, &memreq);
    auto memflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    interop.image_memory = allocateMemory(memreq, memflags, nullptr);
    deletors.add([=,this]{
        vkFreeMemory(logical_device, interop.image_memory, nullptr);
    });
    vkBindImageMemory(logical_device, interop.image, interop.image_memory, 0);

    auto view_type = getImageViewType(params.layout);
    auto info = vkinit::imageViewCreateInfo(interop.image,
        view_type, interop.vk_format, VK_IMAGE_ASPECT_COLOR_BIT
    );
    validation::checkVulkan(vkCreateImageView(logical_device, &info, nullptr, &interop.vk_view));
    deletors.add([=,this]{
        vkDestroyImageView(logical_device, interop.vk_view, nullptr);
    });
    transitionImageLayout(interop.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );
}

void InteropDevice::initViewImage(InteropView& view)
{
    const auto &params = view.params;
    if (params.domain_type == DomainType::Structured)
    {
        const std::vector<Vertex> vertices{
            { {  1.f,  1.f, 0.f }, { 1.f, 1.f } },
            { { -1.f,  1.f, 0.f }, { 0.f, 1.f } },
            { { -1.f, -1.f, 0.f }, { 0.f, 0.f } },
            { {  1.f, -1.f, 0.f }, { 1.f, 0.f } }
        };
        // Indices for a single uv-view quad made from two triangles
        const std::vector<uint16_t> indices{ 0, 1, 2, 2, 3, 0 };//, 4, 5, 6, 6, 7, 4 };

        uint32_t vert_size = sizeof(Vertex) * vertices.size();
        uint32_t ids_size = sizeof(uint16_t) * indices.size();

        // Auxiliary buffer for holding a quad and its indices
        auto usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        view.aux_buffer = createBuffer(vert_size + ids_size, usage);
        VkMemoryRequirements memreq{};
        vkGetBufferMemoryRequirements(logical_device, view.aux_buffer, &memreq);

        // Allocate memory and bind it to buffers
        auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        view.aux_memory = allocateMemory(memreq, flags);
        vkBindBufferMemory(logical_device, view.aux_buffer, view.aux_memory, 0);

        // Init image quad coords and indices
        char *data = nullptr;
        vkMapMemory(logical_device, view.aux_memory, 0, vert_size, 0, (void**)&data);
        std::memcpy(data, vertices.data(), vert_size);
        std::memcpy(data + vert_size, indices.data(), ids_size);
        vkUnmapMemory(logical_device, view.aux_memory);
        deletors.add([=,this]{
            vkDestroyBuffer(logical_device, view.aux_buffer, nullptr);
            vkFreeMemory(logical_device, view.aux_memory, nullptr);
        });

        view.vert_buffers.push_back(view.aux_buffer);
        view.buffer_offsets.push_back(0);
        view.index_offset = 4 * sizeof(Vertex);
        view.idx_type = VK_INDEX_TYPE_UINT16;
    }

    /*for (const auto &[attr, memory] : view.params.attributes)
    {
        view.vert_buffers.push_back(memory.data_buffer);
        view.buffer_offsets.push_back(0);
    }*/
}

void InteropDevice::updateLinearTexture(InteropMemory &interop)
{
    transitionImageLayout(interop.image,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    );

    copyBufferToTexture(interop.data_buffer, interop.image, interop.vk_extent);

    transitionImageLayout(interop.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );
}

void InteropDevice::loadTexture(InteropMemory *interop, void *img_data)
{
    constexpr int level_count = 1;
    auto params = interop->params;
    auto sz = getSize(params.element_count, params.layout);
    size_t image_width  = sz.x;
    size_t image_height = sz.y;

    // Create staging buffer to copy image data
    VkDeviceSize staging_size = image_width * image_height * 4;
    auto staging_buffer = createBuffer(staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    VkMemoryRequirements staging_req;
    vkGetBufferMemoryRequirements(logical_device, staging_buffer, &staging_req);
    auto staging_memory = allocateMemory(staging_req,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    vkBindBufferMemory(logical_device, staging_buffer, staging_memory, 0);

    const auto element_size = getDataSize(params.data_type, params.channel_count);
    const auto element_count = getElementCount(params.element_count, params.layout);
    VkDeviceSize memsize = element_size * element_count;
    char *data = nullptr;
    vkMapMemory(logical_device, staging_memory, 0, memsize, 0, (void**)&data);
    memcpy(data, img_data, static_cast<size_t>(memsize));
    vkUnmapMemory(logical_device, staging_memory);

    transitionImageLayout(interop->image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    );
    copyBufferToTexture(staging_buffer, interop->image, interop->vk_extent);

    vkDestroyBuffer(logical_device, staging_buffer, nullptr);
    vkFreeMemory(logical_device, staging_memory, nullptr);

    generateMipmaps(interop->image, interop->vk_format, image_width, image_height, level_count);

    // TODO: Handle this properly
    validation::checkCuda(cudaDeviceSynchronize());
}

void InteropDevice::copyBufferToTexture(VkBuffer buffer, VkImage image, VkExtent3D extent)
{
    VkImageSubresourceLayers subres;
    subres.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    subres.mipLevel       = 0;
    subres.baseArrayLayer = 0;
    subres.layerCount     = 1;

    VkBufferImageCopy region{};
    region.bufferOffset      = 0;
    region.bufferRowLength   = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource  = subres;
    region.imageOffset       = {0, 0, 0};
    region.imageExtent       = extent;
    immediateSubmit([=](VkCommandBuffer cmd)
    {
        vkCmdCopyBufferToImage(cmd, buffer, image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
        );
    });
}

cudaExternalMemory_t InteropDevice::importCudaExternalMemory(
    VkDeviceMemory vk_mem, VkDeviceSize size)
{
    cudaExternalMemoryHandleDesc extmem_desc{};
    extmem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    extmem_desc.size = size;
    extmem_desc.handle.fd = (int)(uintptr_t)getMemoryHandle(
        vk_mem, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    );
    cudaExternalMemory_t cuda_mem;
    validation::checkCuda(cudaImportExternalMemory(&cuda_mem, &extmem_desc));
    return cuda_mem;
}

void *InteropDevice::getMemoryHandle(VkDeviceMemory memory,
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

void *InteropDevice::getSemaphoreHandle(VkSemaphore semaphore,
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

InteropBarrier InteropDevice::createInteropBarrier()
{
    VkSemaphoreTypeCreateInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timeline_info.pNext = nullptr;
    timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_info.initialValue = 0;

    auto handle_type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkExportSemaphoreCreateInfoKHR export_info{};
    export_info.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
    export_info.pNext       = &timeline_info;
    export_info.handleTypes = handle_type;

    InteropBarrier barrier;
    barrier.vk_semaphore = createSemaphore(&export_info);

    cudaExternalSemaphoreHandleDesc desc{};
    desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    desc.handle.fd = (int)(uintptr_t)getSemaphoreHandle(barrier.vk_semaphore, handle_type);
    desc.flags = 0;
    validation::checkCuda(cudaImportExternalSemaphore(&barrier.cuda_semaphore, &desc));
    deletors.add([=]{
        validation::checkCuda(cudaDestroyExternalSemaphore(barrier.cuda_semaphore));
    });

    return barrier;
}

} // namespace mimir