#include <mimir/engine/interop_device.hpp>

#include <cuda_runtime.h> // make_cudaExtent

#include <cstring> // std::memcpy

#include <mimir/shader_types.hpp>
#include <mimir/validation.hpp>
#include "internal/vk_initializers.hpp"

namespace mimir
{

VkBufferUsageFlags getUsageFlags(ElementType p)
{
    if (p == ElementType::Texels) return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    switch (p)
    {
        case ElementType::Markers: case ElementType::Voxels:
        {
            return usage | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        }
        case ElementType::Edges:
        {
            return usage | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        }
        default: return usage;
    }
}

// Converts a InteropView image type to its Vulkan equivalent
VkImageType getImageType(DataDomain domain)
{
    switch (domain)
    {
        case DataDomain::Domain2D: return VK_IMAGE_TYPE_2D;
        case DataDomain::Domain3D: return VK_IMAGE_TYPE_3D;
        default:                   return VK_IMAGE_TYPE_1D;
    }
}

// Converts a InteropView domain type to its Vulkan equivalent
VkImageViewType getViewType(DataDomain domain)
{
    switch (domain)
    {
        case DataDomain::Domain2D: return VK_IMAGE_VIEW_TYPE_2D;
        case DataDomain::Domain3D: return VK_IMAGE_VIEW_TYPE_3D;
        default:                   return VK_IMAGE_VIEW_TYPE_1D;
    }
}

VkImageTiling getImageTiling(ResourceType type)
{
    switch (type)
    {
        case ResourceType::Texture: return VK_IMAGE_TILING_OPTIMAL;
        default:                    return VK_IMAGE_TILING_LINEAR;
    }
}

void InteropDevice::initTextureQuad(VkBuffer& buf, VkDeviceMemory& mem)
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
    buf = createBuffer(vert_size + ids_size, usage);
    VkMemoryRequirements memreq{};
    vkGetBufferMemoryRequirements(logical_device, buf, &memreq);

    // Allocate memory and bind it to buffers
    auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    mem = allocateMemory(memreq, flags);

    vkBindBufferMemory(logical_device, buf, mem, 0);
    char *data = nullptr;
    vkMapMemory(logical_device, mem, 0, vert_size, 0, (void**)&data);
    std::memcpy(data, vertices.data(), vert_size);
    std::memcpy(data + vert_size, indices.data(), ids_size);
    vkUnmapMemory(logical_device, mem);
}

void initImplicitCoords(VkDevice dev, VkDeviceMemory mem, VkDeviceSize memsize, VkExtent3D extent)
{
    float3 *data = nullptr;
    vkMapMemory(dev, mem, 0, memsize, 0, (void**)&data);
    auto slice_size = extent.width * extent.height;
    for (uint32_t z = 0; z < extent.depth; ++z)
    {
        auto rz = static_cast<float>(z) / extent.depth;
        for (uint32_t y = 0; y < extent.height; ++y)
        {
            auto ry = static_cast<float>(y) / extent.height;
            for (uint32_t x = 0; x < extent.width; ++x)
            {
                auto rx = static_cast<float>(x) / extent.width;
                data[slice_size * z + extent.width * y + x] = float3{rx, ry, rz};
            }
        }
    }
    vkUnmapMemory(dev, mem);
}

VkImage createImage(VkDevice dev, ViewParams params)
{
    auto img_type = getImageType(params.data_domain);
    auto format = getDataFormat(params.data_type, params.channel_count);
    VkExtent3D extent = {params.extent.x, params.extent.y, params.extent.z};
    auto tiling = getImageTiling(params.resource_type);
    auto usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    VkExternalMemoryImageCreateInfo extmem_info{};
    extmem_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    extmem_info.pNext = nullptr;
    extmem_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    // TODO: Check if texture is within bounds
    //auto max_img_dim = properties.limits.maxImageDimension3D;

    auto info = vkinit::imageCreateInfo(img_type, format, extent, usage);
    info.pNext         = &extmem_info;
    info.flags         = 0;
    info.tiling        = tiling;
    info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateImage(dev, &info, nullptr, &image));
    return image;
}

cudaMipmappedArray_t createMipmapArray(cudaExternalMemory_t cuda_extmem, ViewParams params)
{
    constexpr int level_count = 1; // TODO: Should be a parameter

    cudaChannelFormatDesc format_desc;
    format_desc.x = 8;
    format_desc.y = 8;
    format_desc.z = 8;
    format_desc.w = 8;
    format_desc.f = cudaChannelFormatKindUnsigned;
    size_t image_width  = params.extent.x;
    size_t image_height = params.extent.y;
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

void InteropDevice::initView(InteropView& view)
{
    const auto params = view.params;
    const auto element_size = getDataSize(params.data_type, params.channel_count);
    view.vk_format = getDataFormat(params.data_type, params.channel_count);
    view.vk_extent = {params.extent.x, params.extent.y, params.extent.z};

    VkMemoryRequirements memreq{};
    if (params.resource_type == ResourceType::Buffer)
    {
        if (params.domain_type == DomainType::Structured)
        {
            // Allocate memory and bind it to buffers
            auto buffer_size = sizeof(float3) * params.element_count;
            view.aux_buffer = createBuffer(buffer_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            VkMemoryRequirements memreq;
            vkGetBufferMemoryRequirements(logical_device, view.aux_buffer, &memreq);
            auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            view.aux_memory = allocateMemory(memreq, flags);
            deletors.add([=,this]{
                vkDestroyBuffer(logical_device, view.aux_buffer, nullptr);
                vkFreeMemory(logical_device, view.aux_memory, nullptr);
            });            
            vkBindBufferMemory(logical_device, view.aux_buffer, view.aux_memory, 0);

            initImplicitCoords(logical_device, view.aux_memory, buffer_size, view.vk_extent);
        }

        // Create view buffers
        VkDeviceSize memsize = element_size * params.element_count;
        auto usage = getUsageFlags(params.element_type);
        VkExternalMemoryBufferCreateInfo extmem_info{};
        extmem_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        extmem_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        view.data_buffer = createBuffer(memsize, usage, &extmem_info);
        deletors.add([=,this]{
            vkDestroyBuffer(logical_device, view.data_buffer, nullptr);
        });
        vkGetBufferMemoryRequirements(logical_device, view.data_buffer, &memreq);
    }
    if (params.resource_type == ResourceType::Texture || params.element_type == ElementType::Texels)
    {
        // Init texture memory
        view.image = createImage(logical_device, params);
        view.vk_sampler = createSampler(VK_FILTER_NEAREST, true);
        deletors.add([=,this]{
            vkDestroyImage(logical_device, view.image, nullptr);
        });

        vkGetImageMemoryRequirements(logical_device, view.image, &memreq);
    }

    VkExportMemoryAllocateInfoKHR export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    export_info.pNext = nullptr;
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    auto properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    view.memory = allocateMemory(memreq, properties, &export_info);
    view.cuda_extmem = importCudaExternalMemory(view.memory, memreq.size);
    deletors.add([=,this]{
        validation::checkCuda(cudaDestroyExternalMemory(view.cuda_extmem));
        vkFreeMemory(logical_device, view.memory, nullptr);
    });

    if (params.resource_type == ResourceType::Buffer)
    {
        vkBindBufferMemory(logical_device, view.data_buffer, view.memory, 0);
        VkDeviceSize memsize = element_size * params.element_count;
        cudaExternalMemoryBufferDesc buffer_desc{};
        buffer_desc.offset = 0;
        buffer_desc.size   = memsize;
        buffer_desc.flags  = 0;
        validation::checkCuda(cudaExternalMemoryGetMappedBuffer(
            &view.cuda_ptr, view.cuda_extmem, &buffer_desc)
        );
    }
    if (params.resource_type == ResourceType::Texture || params.element_type == ElementType::Texels)
    {
        initTextureQuad(view.aux_buffer, view.aux_memory);
        deletors.add([=,this]{
            vkDestroyBuffer(logical_device, view.aux_buffer, nullptr);
            vkFreeMemory(logical_device, view.aux_memory, nullptr);
        });

        view.index_offset = 4 * sizeof(Vertex);
        vkBindImageMemory(logical_device, view.image, view.memory, 0);

        auto view_type = getViewType(params.data_domain);
        auto info = vkinit::imageViewCreateInfo(view.image,
            view_type, view.vk_format, VK_IMAGE_ASPECT_COLOR_BIT
        );
        validation::checkVulkan(vkCreateImageView(logical_device, &info, nullptr, &view.vk_view));
        deletors.add([=,this]{
            vkDestroyImageView(logical_device, view.vk_view, nullptr);
        });

        // Init texture memory (TODO: Refactor)
        if (params.element_type == ElementType::Texels)
        {
            VkDeviceSize memsize = element_size * params.element_count;
            cudaExternalMemoryBufferDesc buffer_desc{};
            buffer_desc.offset = 0;
            buffer_desc.size   = memsize;
            buffer_desc.flags  = 0;
            validation::checkCuda(cudaExternalMemoryGetMappedBuffer(
                &view.cuda_ptr, view.cuda_extmem, &buffer_desc)
            );
            transitionImageLayout(view.image,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            );
        }
        else if (params.resource_type == ResourceType::Texture)
        {
            view.mipmap_array = createMipmapArray(view.cuda_extmem, params);
            deletors.add([=,this]{
                validation::checkCuda(cudaFreeMipmappedArray(view.mipmap_array));
            });
        }
    }
}

void InteropDevice::updateTexture(InteropView& view)
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
    immediateSubmit([=,this](VkCommandBuffer cmd)
    {
        vkCmdCopyBufferToImage(cmd, view.data_buffer, view.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
        );
    });

    transitionImageLayout(view.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );
}

void InteropDevice::loadTexture(InteropView *view, void *img_data)
{
    auto params = view->params;
    constexpr int level_count = 1;
    size_t image_width  = params.extent.x;
    size_t image_height = params.extent.y;

    // Create staging buffer to copy image data
    VkDeviceSize staging_size = image_width * image_height * 4;
    auto staging_buffer = createBuffer(staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    VkMemoryRequirements staging_req;
    vkGetBufferMemoryRequirements(logical_device, staging_buffer, &staging_req);
    auto staging_memory = allocateMemory(staging_req,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    vkBindBufferMemory(logical_device, staging_buffer, staging_memory, 0);

    char *data = nullptr;
    VkDeviceSize memsize = getDataSize(params.data_type, params.channel_count) * params.element_count;
    vkMapMemory(logical_device, staging_memory, 0, memsize, 0, (void**)&data);
    memcpy(data, img_data, static_cast<size_t>(memsize));
    vkUnmapMemory(logical_device, staging_memory);

    transitionImageLayout(view->image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    );

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
    region.imageExtent       = view->vk_extent;
    immediateSubmit([=,this](VkCommandBuffer cmd)
    {
        vkCmdCopyBufferToImage(cmd, staging_buffer, view->image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
        );
    });
    vkDestroyBuffer(logical_device, staging_buffer, nullptr);
    vkFreeMemory(logical_device, staging_memory, nullptr);

    generateMipmaps(view->image, view->vk_format, image_width, image_height, level_count);

    // TODO: Handle this properly
    validation::checkCuda(cudaDeviceSynchronize());
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
    deletors.add([=,this]{
        validation::checkCuda(cudaDestroyExternalSemaphore(barrier.cuda_semaphore));
    });

    return barrier;
}

} // namespace mimir