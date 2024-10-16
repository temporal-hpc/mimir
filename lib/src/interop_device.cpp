#include <mimir/engine/interop_device.hpp>

#include <cstring> // std::memcpy

#include "internal/shader_types.hpp"
#include "internal/resources.hpp"
#include "internal/validation.hpp"

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

// Converts a InteropViewOld image type to its Vulkan equivalent
VkImageType getImageType(DataLayout layout)
{
    switch (layout)
    {
        case DataLayout::Layout2D: return VK_IMAGE_TYPE_2D;
        case DataLayout::Layout3D: return VK_IMAGE_TYPE_3D;
        default:                   return VK_IMAGE_TYPE_1D;
    }
}

// Converts a InteropViewOld layout type to its Vulkan equivalent
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

uint32_t InteropDevice::getMaxImageDimension(DataLayout layout)
{
    auto limits = physical_device.general.properties.limits;
    switch (layout)
    {
        case DataLayout::Layout1D: return limits.maxImageDimension1D;
        case DataLayout::Layout2D: return limits.maxImageDimension2D;
        case DataLayout::Layout3D: return limits.maxImageDimension3D;
        default: return 0;
    }
}

VkImage InteropDevice::createImage2(MemoryParams mp)
{
    auto sz = mp.element_count;
    ImageParams params{
        .type   = getImageType(mp.layout),
        .format = getDataFormat(mp.component_type, mp.channel_count),
        .extent = {sz.x, sz.y, sz.z},
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    };

    return createImage(logical_device, physical_device.handle, params);
}

void InteropDevice::initMemoryBuffer(InteropMemory& interop)
{
    const auto &params = interop.params;
    const auto element_size = getBytesize(params.component_type, params.channel_count);
    const auto element_count = getElementCount(params.element_count, params.layout);

    // Create external memory buffers
    VkDeviceSize memsize = element_size * element_count;
    auto usage = getBufferUsage(params.resource_type);
    VkExternalMemoryBufferCreateInfo extmem_info{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };

    interop.data_buffer = createBuffer(logical_device, memsize, usage, &extmem_info);
    VkMemoryRequirements memreq{};
    vkGetBufferMemoryRequirements(logical_device, interop.data_buffer, &memreq);

    // Create and export (to CUDA) the memory allocated with the
    // requirements obtained above
    VkExportMemoryAllocateInfoKHR export_info{
        .sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    auto available = physical_device.memory.memoryProperties;
    auto memflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    interop.memory = allocateMemory(logical_device, available, memreq, memflags, &export_info);
    interop.cuda_extmem = interop::importCudaExternalMemory(
        interop.memory, memreq.size, logical_device
    );

    // Bind the resources to the external memory allocated above
    vkBindBufferMemory(logical_device, interop.data_buffer, interop.memory, 0);
    cudaExternalMemoryBufferDesc buffer_desc{
        .offset = 0,
        .size   = memsize,
        .flags  = 0
    };
    validation::checkCuda(cudaExternalMemoryGetMappedBuffer(
        &interop.cuda_ptr, interop.cuda_extmem, &buffer_desc)
    );
}

void InteropDevice::initViewBuffer(InteropViewOld& view)
{
    const auto &params = view.params;
    // For structured domain views, initialize auxiliary resources and memory
    if (params.domain_type == DomainType::Structured)
    {
        // Allocate memory and bind it to buffers
        auto buffer_size = sizeof(float3) * params.element_count;
        view.aux_buffer = createBuffer(logical_device, buffer_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        VkMemoryRequirements memreq{};
        vkGetBufferMemoryRequirements(logical_device, view.aux_buffer, &memreq);
        auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        auto available = physical_device.memory.memoryProperties;
        view.aux_memory = allocateMemory(logical_device, available, memreq, flags);
        vkBindBufferMemory(logical_device, view.aux_buffer, view.aux_memory, 0);

        //initImplicitCoords(logical_device, view.aux_memory, buffer_size, params.extent);
        view.vert_buffers.push_back(view.aux_buffer);
        view.buffer_offsets.push_back(0);
    }

    for (const auto &[attr, memory] : view.params.attributes)
    {
        if (attr == AttributeType::Index)
        {
            view.idx_buffer = memory.data_buffer;
            view.idx_type = getIndexType(memory.params.component_type);
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
    auto sz = params.element_count;
    interop.vk_extent = {sz.x, sz.y, sz.z};
    interop.vk_format = getDataFormat(params.component_type, params.channel_count);

    // Init texture memory
    interop.image = createImage2(params);
    interop.vk_sampler = createSampler(logical_device, VK_FILTER_NEAREST, true);
    VkMemoryRequirements memreq{};
    vkGetImageMemoryRequirements(logical_device, interop.image, &memreq);

    auto memflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkExportMemoryAllocateInfoKHR export_info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    auto available = physical_device.memory.memoryProperties;
    interop.image_memory = allocateMemory(logical_device, available, memreq, memflags, &export_info);
    interop.cuda_extmem = interop::importCudaExternalMemory(
        interop.image_memory, memreq.size, logical_device
    );

    vkBindImageMemory(logical_device, interop.image, interop.image_memory, 0);

    VkImageViewCreateInfo info{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .image    = interop.image,
        .viewType = getImageViewType(params.layout),
        .format   = interop.vk_format,
        // Default mapping of all color channels
        .components = VkComponentMapping{
            .r = VK_COMPONENT_SWIZZLE_R,
            .g = VK_COMPONENT_SWIZZLE_G,
            .b = VK_COMPONENT_SWIZZLE_B,
            .a = VK_COMPONENT_SWIZZLE_A,
        },
        // Describe image purpose and which part of it should be accesssed
        .subresourceRange = VkImageSubresourceRange{
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        }
    };
    validation::checkVulkan(vkCreateImageView(logical_device, &info, nullptr, &interop.vk_view));

    transitionImageLayout(interop.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    int size = getComponentSize(params.component_type) * 8;
    int4 comp_sz = {size, size, size, size};
    int3 extent  = {(int)params.element_count.x, (int)params.element_count.y, 0};
    interop.mipmap_array = interop::createMipmapArray(
        interop.cuda_extmem, comp_sz, extent, 1
    );
}

void InteropDevice::initMemoryImageLinear(InteropMemory& interop)
{
    initMemoryBuffer(interop);

    const auto &params = interop.params;
    auto sz = params.element_count;
    interop.vk_extent = {sz.x, sz.y, sz.z};
    interop.vk_format = getDataFormat(params.component_type, params.channel_count);

    // Init texture memory
    interop.image = createImage2(params);
    interop.vk_sampler = createSampler(logical_device, VK_FILTER_NEAREST, true);

    VkMemoryRequirements memreq{};
    vkGetImageMemoryRequirements(logical_device, interop.image, &memreq);
    auto memflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    auto available = physical_device.memory.memoryProperties;
    interop.image_memory = allocateMemory(logical_device, available, memreq, memflags, nullptr);
    vkBindImageMemory(logical_device, interop.image, interop.image_memory, 0);

    VkImageViewCreateInfo info{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .image    = interop.image,
        .viewType = getImageViewType(params.layout),
        .format   = interop.vk_format,
        // Default mapping of all color channels
        .components = VkComponentMapping{
            .r = VK_COMPONENT_SWIZZLE_R,
            .g = VK_COMPONENT_SWIZZLE_G,
            .b = VK_COMPONENT_SWIZZLE_B,
            .a = VK_COMPONENT_SWIZZLE_A,
        },
        // Describe image purpose and which part of it should be accesssed
        .subresourceRange = VkImageSubresourceRange{
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        }
    };
    validation::checkVulkan(vkCreateImageView(logical_device, &info, nullptr, &interop.vk_view));
    transitionImageLayout(interop.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );
}

void InteropDevice::initViewImage(InteropViewOld& view)
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
        view.aux_buffer = createBuffer(logical_device, vert_size + ids_size, usage);
        VkMemoryRequirements memreq{};
        vkGetBufferMemoryRequirements(logical_device, view.aux_buffer, &memreq);

        // Allocate memory and bind it to buffers
        auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        auto available = physical_device.memory.memoryProperties;
        view.aux_memory = allocateMemory(logical_device, available, memreq, flags);
        vkBindBufferMemory(logical_device, view.aux_buffer, view.aux_memory, 0);

        // Init image quad coords and indices
        char *data = nullptr;
        vkMapMemory(logical_device, view.aux_memory, 0, vert_size, 0, (void**)&data);
        std::memcpy(data, vertices.data(), vert_size);
        std::memcpy(data + vert_size, indices.data(), ids_size);
        vkUnmapMemory(logical_device, view.aux_memory);

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
    size_t image_width  = params.element_count.x;
    size_t image_height = params.element_count.y;

    // Create staging buffer to copy image data
    VkDeviceSize staging_size = image_width * image_height * 4;
    auto staging_buffer = createBuffer(logical_device, staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    VkMemoryRequirements staging_req;
    vkGetBufferMemoryRequirements(logical_device, staging_buffer, &staging_req);
    auto available = physical_device.memory.memoryProperties;
    auto staging_memory = allocateMemory(logical_device, available, staging_req,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    vkBindBufferMemory(logical_device, staging_buffer, staging_memory, 0);

    const auto element_size = getBytesize(params.component_type, params.channel_count);
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
    VkImageSubresourceLayers subres{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1
    };
    VkBufferImageCopy region{
        .bufferOffset      = 0,
        .bufferRowLength   = 0,
        .bufferImageHeight = 0,
        .imageSubresource  = subres,
        .imageOffset       = {0, 0, 0},
        .imageExtent       = extent
    };
    immediateSubmit([=](VkCommandBuffer cmd)
    {
        auto layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        vkCmdCopyBufferToImage(cmd, buffer, image, layout, 1, &region);
    });
}

} // namespace mimir