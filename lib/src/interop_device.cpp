#include <mimir/engine/interop_device.hpp>

#include <cstring> // std::memcpy

#include <mimir/engine/shader_types.hpp>
#include "internal/validation.hpp"
#include "internal/interop.hpp"

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

VkImage InteropDevice::createImage(MemoryParams params)
{
    auto img_type = getImageType(params.layout);
    auto format   = getDataFormat(params.component_type, params.channel_count);
    auto tiling   = VK_IMAGE_TILING_OPTIMAL;
    auto sz = params.element_count;
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkExtent3D extent = {sz.x, sz.y, sz.z};

    // TODO: Check if texture is within bounds
    auto max_dim = getMaxImageDimension(params.layout);
    if (extent.width >= max_dim || extent.height >= max_dim || extent.height >= max_dim)
    {
        spdlog::error("Requested image dimensions are larger than maximum");
    }

    // Check that the upcoming image parameters are supported
    VkPhysicalDeviceImageFormatInfo2 format_info{
        .sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
        .pNext  = nullptr,
        .format = format,
        .type   = img_type,
        .tiling = tiling,
        .usage  = usage,
        .flags  = 0
    };
    VkImageFormatProperties2 format_props{
        .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
        .pNext = nullptr,
        .imageFormatProperties = {} // To be filled in function below
    };
    // TODO: Do not rely on validation layers to stop invalid image formats
    validation::checkVulkan(vkGetPhysicalDeviceImageFormatProperties2(
        physical_device.handle, &format_info, &format_props
    ));

    VkExternalMemoryImageCreateInfo extmem_info{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkImageCreateInfo info{
        .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext       = &extmem_info,
        .flags       = 0,
        .imageType   = img_type,
        .format      = format,
        .extent      = extent,
        .mipLevels   = 1,
        .arrayLayers = 1,
        .samples     = VK_SAMPLE_COUNT_1_BIT,
        .tiling      = tiling,
        .usage       = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage image = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateImage(logical_device, &info, nullptr, &image));
    return image;
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

    interop.data_buffer = createBuffer(memsize, usage, &extmem_info);
    VkMemoryRequirements memreq{};
    vkGetBufferMemoryRequirements(logical_device, interop.data_buffer, &memreq);

    // Create and export (to CUDA) the memory allocated with the
    // requirements obtained above
    VkExportMemoryAllocateInfoKHR export_info{
        .sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    auto memflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    interop.memory = allocateMemory(memreq, memflags, &export_info);
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
        view.aux_buffer = createBuffer(buffer_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        VkMemoryRequirements memreq{};
        vkGetBufferMemoryRequirements(logical_device, view.aux_buffer, &memreq);
        auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        view.aux_memory = allocateMemory(memreq, flags);
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
    interop.image = createImage(params);
    interop.vk_sampler = createSampler(VK_FILTER_NEAREST, true);
    VkMemoryRequirements memreq{};
    vkGetImageMemoryRequirements(logical_device, interop.image, &memreq);

    auto memflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkExportMemoryAllocateInfoKHR export_info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    interop.image_memory = allocateMemory(memreq, memflags, &export_info);
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
    interop.image = createImage(params);
    interop.vk_sampler = createSampler(VK_FILTER_NEAREST, true);

    VkMemoryRequirements memreq{};
    vkGetImageMemoryRequirements(logical_device, interop.image, &memreq);
    auto memflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    interop.image_memory = allocateMemory(memreq, memflags, nullptr);
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
    auto staging_buffer = createBuffer(staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    VkMemoryRequirements staging_req;
    vkGetBufferMemoryRequirements(logical_device, staging_buffer, &staging_req);
    auto staging_memory = allocateMemory(staging_req,
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

InteropBarrier InteropDevice::createInteropBarrier()
{
    VkSemaphoreTypeCreateInfo timeline_info{
        .sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext         = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue  = 0
    };
    VkExportSemaphoreCreateInfoKHR export_info{
        .sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR,
        .pNext       = &timeline_info,
        .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    InteropBarrier barrier;
    barrier.vk_semaphore = createSemaphore(&export_info);
    barrier.cuda_semaphore = interop::importCudaExternalSemaphore(barrier.vk_semaphore, logical_device);
    return barrier;
}

} // namespace mimir