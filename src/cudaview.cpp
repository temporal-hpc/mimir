#include <cudaview/engine/cudaview.hpp>

#include <cuda_runtime.h>
#include <cstring> // std::memcpy

#include "cudaview/vk_types.hpp"
#include "internal/vk_initializers.hpp"
#include "internal/utils.hpp"
#include <cudaview/validation.hpp>

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

// Converts a CudaView texture type to its Vulkan equivalent
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

// Converts a CudaView image type to its Vulkan equivalent
VkImageType getImageType(DataDomain domain)
{
    switch (domain)
    {
        case DataDomain::Domain2D: return VK_IMAGE_TYPE_2D;
        case DataDomain::Domain3D: return VK_IMAGE_TYPE_3D;
        default:                   return VK_IMAGE_TYPE_1D;
    }
}

// Converts a CudaView domain type to its Vulkan equivalent
VkImageViewType getViewType(DataDomain domain)
{
    switch (domain)
    {
        case DataDomain::Domain2D: return VK_IMAGE_VIEW_TYPE_2D;
        case DataDomain::Domain3D: return VK_IMAGE_VIEW_TYPE_3D;
        default:                   return VK_IMAGE_VIEW_TYPE_1D;
    }
}

InteropMemory getInteropImage(ViewParams params, VulkanCudaDevice *dev)
{
    constexpr int level_count = 1; // TODO: Should be a parameter
    InteropMemory interop;

    // Init texture memory
    auto img_type = getImageType(params.data_domain);
    VkExternalMemoryImageCreateInfo ext_info{};
    ext_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    ext_info.pNext = nullptr;
    ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    auto img_format = getVulkanFormat(params.texture_format);
    VkExtent3D img_extent = {params.extent.x, params.extent.y, params.extent.z};
    interop.image = dev->createImage(img_type, img_format, img_extent,
        VK_IMAGE_TILING_OPTIMAL, 
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        &ext_info
    );

    VkExportMemoryAllocateInfoKHR export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    export_info.pNext = nullptr;
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryRequirements reqs;
    vkGetImageMemoryRequirements(dev->logical_device, interop.image, &reqs);
    interop.memory = dev->allocateMemory(reqs,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &export_info
    );

    vkBindImageMemory(dev->logical_device, interop.image, interop.memory, 0);
    dev->importCudaExternalMemory(interop.cuda_extmem, interop.memory, reqs.size);
    
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

    validation::checkCuda(cudaExternalMemoryGetMappedMipmappedArray(
        &interop.mipmap_array, interop.cuda_extmem, &array_desc)
    );

    dev->deletors.pushFunction([=]{
        validation::checkCuda(cudaFreeMipmappedArray(interop.mipmap_array));
        validation::checkCuda(cudaDestroyExternalMemory(interop.cuda_extmem));
        vkDestroyImage(dev->logical_device, interop.image, nullptr);
        vkFreeMemory(dev->logical_device, interop.memory, nullptr);
    });

    return interop;
}

InteropMemory getInteropBuffer(ViewParams params, VulkanCudaDevice *dev)
{
    InteropMemory interop;

    VkDeviceSize memsize = params.element_size * params.element_count;
    auto usage = getUsageFlags(params.primitive_type, params.resource_type);

    // Create interop buffers
    VkExternalMemoryBufferCreateInfo extmem_info{};
    extmem_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extmem_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    interop.data_buffer = dev->createBuffer(memsize, usage, &extmem_info);

    VkExportMemoryAllocateInfoKHR export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    export_info.pNext = nullptr;
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(dev->logical_device, interop.data_buffer, &reqs);
    interop.memory = dev->allocateMemory(reqs,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &export_info
    );

    vkBindBufferMemory(dev->logical_device, interop.data_buffer, interop.memory, 0);
    dev->importCudaExternalMemory(interop.cuda_extmem, interop.memory, memsize);
    cudaExternalMemoryBufferDesc buffer_desc{};
    buffer_desc.offset = 0;
    buffer_desc.size   = memsize;
    buffer_desc.flags  = 0;
    validation::checkCuda(cudaExternalMemoryGetMappedBuffer(
        &interop.cuda_ptr, interop.cuda_extmem, &buffer_desc)
    );
    
    dev->deletors.pushFunction([=]{
        validation::checkCuda(cudaDestroyExternalMemory(interop.cuda_extmem));
        vkDestroyBuffer(dev->logical_device, interop.data_buffer, nullptr);
        vkFreeMemory(dev->logical_device, interop.memory, nullptr);
    }); 

    return interop;
}

CudaView::CudaView(ViewParams params, VulkanCudaDevice *dev): _params{params}, _dev{dev}
{}

void CudaView::init()
{
    vk_format = getVulkanFormat(_params.texture_format);
    vk_extent = {_params.extent.x, _params.extent.y, _params.extent.z};

    auto usage = getUsageFlags(_params.primitive_type, _params.resource_type);
    VkMemoryRequirements requirements;
    auto logical_device = _dev->logical_device;
    
    if (_params.resource_type == ResourceType::Texture || 
        _params.resource_type == ResourceType::TextureLinear)
    {
        const std::vector<Vertex> vertices{
        { {  1.f,  1.f, 0.f }, { 1.f, 1.f } },
        { { -1.f,  1.f, 0.f }, { 0.f, 1.f } },
        { { -1.f, -1.f, 0.f }, { 0.f, 0.f } },
        { {  1.f, -1.f, 0.f }, { 1.f, 0.f } }
        };
        // Indices for a single uv-view quad made from two triangles
        const std::vector<uint16_t> indices{ 0, 1, 2, 2, 3, 0};//, 4, 5, 6, 6, 7, 4 };

        uint32_t vert_size = sizeof(Vertex) * vertices.size();
        uint32_t ids_size = sizeof(uint16_t) * indices.size();

        // Test buffer for asking about its memory properties
        usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        auto requirements = _dev->getMemoryRequiements(usage, {vert_size, ids_size});
        auto max_aligned_size = requirements.size;
        // The largest alignment requirement can be used for all
        requirements.size = max_aligned_size * 2;

        // Allocate memory and bind it to buffers
        aux_memory = _dev->allocateMemory(requirements,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        vertex_buffer = _dev->createBuffer(vert_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        vkBindBufferMemory(logical_device, vertex_buffer, aux_memory, 0);
        char *data = nullptr;
        vkMapMemory(logical_device, aux_memory, 0, vert_size, 0, (void**)&data);
        std::memcpy(data, vertices.data(), vert_size);
        vkUnmapMemory(logical_device, aux_memory);

        index_buffer = _dev->createBuffer(ids_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        vkBindBufferMemory(logical_device, index_buffer, aux_memory, max_aligned_size);
        data = nullptr;
        vkMapMemory(logical_device, aux_memory, max_aligned_size, ids_size, 0, (void**)&data);
        std::memcpy(data, indices.data(), ids_size);
        vkUnmapMemory(logical_device, aux_memory);

        _dev->deletors.pushFunction([=]{
            vkDestroyBuffer(logical_device, vertex_buffer, nullptr);
            vkDestroyBuffer(logical_device, index_buffer, nullptr);
            vkFreeMemory(logical_device, aux_memory, nullptr);
        });

        // Init texture memory (TODO: Refactor)
        if (_params.resource_type == ResourceType::TextureLinear)
        {
            _interop = getInteropBuffer(_params, _dev);
            auto img_type = getImageType(_params.data_domain);
            VkExternalMemoryImageCreateInfo ext_info{};
            ext_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
            ext_info.pNext = nullptr;
            ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

            auto img_format = getVulkanFormat(_params.texture_format);
            VkExtent3D img_extent = {_params.extent.x, _params.extent.y, _params.extent.z};
            _interop.image = _dev->createImage(img_type, img_format, img_extent,
                VK_IMAGE_TILING_LINEAR, 
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                &ext_info
            );
            vkBindImageMemory(logical_device, _interop.image, _interop.memory, 0);
            _dev->transitionImageLayout(_interop.image,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            );
            _dev->deletors.pushFunction([=]{
                vkDestroyImage(logical_device, _interop.image, nullptr);
            });      
        }
        else if (_params.resource_type == ResourceType::Texture)
        {
            _interop = getInteropImage(_params, _dev);
        }

        auto view_type = getViewType(_params.data_domain);
        auto info = vkinit::imageViewCreateInfo(_interop.image,
            view_type, vk_format, VK_IMAGE_ASPECT_COLOR_BIT
        );
        validation::checkVulkan(vkCreateImageView(logical_device, &info, nullptr, &vk_view));
        vk_sampler = _dev->createSampler(VK_FILTER_NEAREST, true);

        _dev->deletors.pushFunction([=]{
            vkDestroyImageView(logical_device, vk_view, nullptr);
        });
    }
    else
    {
        _interop = getInteropBuffer(_params, _dev);
        if (_params.resource_type == ResourceType::StructuredBuffer)
        {
            auto buffer_size = sizeof(float3) * _params.element_count;

            // Test buffer for asking about its memory properties
            auto test_buffer = _dev->createBuffer(1, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            vkGetBufferMemoryRequirements(logical_device, test_buffer, &requirements);
            requirements.size = getAlignedSize(buffer_size, requirements.alignment);
            vkDestroyBuffer(logical_device, test_buffer, nullptr);

            // Allocate memory and bind it to buffers
            aux_memory = _dev->allocateMemory(requirements,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            );
            vertex_buffer = _dev->createBuffer(buffer_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            vkBindBufferMemory(logical_device, vertex_buffer, aux_memory, 0);

            float3 *data = nullptr;
            vkMapMemory(logical_device, aux_memory, 0, buffer_size, 0, (void**)&data);
            auto slice_size = vk_extent.width * vk_extent.height;
            for (uint32_t z = 0; z < vk_extent.depth; ++z)
            {
                auto rz = static_cast<float>(z) / vk_extent.depth;
                for (uint32_t y = 0; y < vk_extent.height; ++y)
                {
                    auto ry = static_cast<float>(y) / vk_extent.height;
                    for (uint32_t x = 0; x < vk_extent.width; ++x)
                    {
                        auto rx = static_cast<float>(x) / vk_extent.width;
                        data[slice_size * z + vk_extent.width * y + x] = float3{rx, ry, rz};
                    }
                }
            }
            vkUnmapMemory(logical_device, aux_memory);

            _dev->deletors.pushFunction([=]{
                vkDestroyBuffer(logical_device, vertex_buffer, nullptr);
                vkFreeMemory(logical_device, aux_memory, nullptr);
            });
        }
    }
}

void CudaView::updateTexture()
{
    _dev->transitionImageLayout(_interop.image,
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
    region.imageExtent = vk_extent;
    _dev->immediateSubmit([=](VkCommandBuffer cmd)
    {
        vkCmdCopyBufferToImage(cmd, _interop.data_buffer, _interop.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
        );
    });

    _dev->transitionImageLayout(_interop.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );
}

void CudaView::loadTexture(void *img_data)
{
    constexpr int level_count = 1;
    size_t image_width  = _params.extent.x;
    size_t image_height = _params.extent.y;

    // Create staging buffer to copy image data
    VkDeviceSize staging_size = image_width * image_height * 4;
    auto staging_buffer = _dev->createBuffer(staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    VkMemoryRequirements staging_req;
    vkGetBufferMemoryRequirements(_dev->logical_device, staging_buffer, &staging_req);
    auto staging_memory = _dev->allocateMemory(staging_req,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    vkBindBufferMemory(_dev->logical_device, staging_buffer, staging_memory, 0);

    char *data = nullptr;
    VkDeviceSize memsize = _params.element_size * _params.element_count;
    vkMapMemory(_dev->logical_device, staging_memory, 0, memsize, 0, (void**)&data);
    memcpy(data, img_data, static_cast<size_t>(memsize));
    vkUnmapMemory(_dev->logical_device, staging_memory);

    _dev->transitionImageLayout(_interop.image,
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
    region.imageExtent       = vk_extent;
    _dev->immediateSubmit([=](VkCommandBuffer cmd)
    {
        vkCmdCopyBufferToImage(cmd, staging_buffer, _interop.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
        );
    });
    vkDestroyBuffer(_dev->logical_device, staging_buffer, nullptr);
    vkFreeMemory(_dev->logical_device, staging_memory, nullptr);

    _dev->generateMipmaps(_interop.image, vk_format, image_width, image_height, level_count);

    // TODO: Handle this properly
    validation::checkCuda(cudaDeviceSynchronize());
}

void CudaView::createUniformBuffers(uint32_t img_count)
{
    auto min_alignment = _dev->properties.limits.minUniformBufferOffsetAlignment;
    auto size_mvp = getAlignedSize(sizeof(ModelViewProjection), min_alignment);
    auto size_options = getAlignedSize(sizeof(PrimitiveParams), min_alignment);
    auto size_scene = getAlignedSize(sizeof(SceneParams), min_alignment);

    VkDeviceSize buffer_size = img_count * (2 * size_mvp + size_options + size_scene);

    auto test_buffer = _dev->createBuffer(1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(_dev->logical_device, test_buffer, &requirements);
    requirements.size = buffer_size;
    vkDestroyBuffer(_dev->logical_device, test_buffer, nullptr);

    // Allocate memory and bind it to buffers
    ubo_memory = _dev->allocateMemory(requirements,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    ubo_buffer = _dev->createBuffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    vkBindBufferMemory(_dev->logical_device, ubo_buffer, ubo_memory, 0);
}

void CudaView::updateUniformBuffers(uint32_t image_idx,
    ModelViewProjection mvp, PrimitiveParams options, SceneParams scene)
{
    auto min_alignment = _dev->properties.limits.minUniformBufferOffsetAlignment;
    auto size_mvp = getAlignedSize(sizeof(ModelViewProjection), min_alignment);
    auto size_options = getAlignedSize(sizeof(PrimitiveParams), min_alignment);
    auto size_scene = getAlignedSize(sizeof(SceneParams), min_alignment);
    auto size_ubo = 2 * size_mvp + size_options + size_scene;
    auto offset = image_idx * size_ubo;

    char *data = nullptr;
    vkMapMemory(_dev->logical_device, ubo_memory, offset, size_ubo, 0, (void**)&data);
    std::memcpy(data, &mvp, sizeof(mvp));
    std::memcpy(data + size_mvp, &options, sizeof(options));
    std::memcpy(data + size_mvp + size_options, &scene, sizeof(scene));
    std::memcpy(data + size_mvp + size_options + size_scene, &mvp, sizeof(mvp));
    vkUnmapMemory(_dev->logical_device, ubo_memory);
}
