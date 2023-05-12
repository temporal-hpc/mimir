#include <cudaview/engine/vk_cudadevice.hpp>

#include <cuda_runtime.h>

#include <cstring> // std::memcpy

#include <cudaview/vk_types.hpp>
#include <cudaview/validation.hpp>
#include "internal/vk_initializers.hpp"

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

VkImageTiling getImageTiling(ResourceType type)
{
    switch (type)
    {
        case ResourceType::Texture: return VK_IMAGE_TILING_OPTIMAL;
        case ResourceType::TextureLinear: default: return VK_IMAGE_TILING_LINEAR;
    }
}

void VulkanCudaDevice::initCuda(uint8_t *device_uuid, size_t uuid_size)
{
    int curr_device = 0, device_count = 0, prohibited_count = 0;

    validation::checkCuda(cudaGetDeviceCount(&device_count));
    if (device_count == 0)
    {
        throw std::runtime_error("No CUDA-enabled device found");
    }

    printf("Enumerating CUDA devices:\n");
    while (curr_device < device_count)
    {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, curr_device);
        printf("* ID: %d\n  Name: %s\n  Capability: %d.%d\n",
            curr_device, dev_prop.name, dev_prop.major, dev_prop.minor
        );
        curr_device++;
    }
    curr_device = 0;
    while (curr_device < device_count)
    {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, curr_device);
        if (dev_prop.computeMode != cudaComputeModeProhibited)
        {
            auto matching = memcmp((void*)&dev_prop.uuid, device_uuid, uuid_size) == 0;
            if (matching)
            {
                validation::checkCuda(cudaSetDevice(curr_device));
                printf("Selected CUDA device %d: %s\n\n", curr_device, dev_prop.name);
            }
        }
        else
        {
            prohibited_count++;
        }

        curr_device++;
    }    

    if (prohibited_count == device_count)
    {
        throw std::runtime_error("No CUDA-Vulkan interop device was found");
    }
}

void VulkanCudaDevice::initView(CudaView& view)
{
    const auto params = view.params;
    view.vk_format = getVulkanFormat(params.texture_format);
    view.vk_extent = {params.extent.x, params.extent.y, params.extent.z};

    VkMemoryRequirements bufreq{}, imgreq{}, memreq{};
    if (params.resource_type == ResourceType::StructuredBuffer || 
        params.resource_type == ResourceType::UnstructuredBuffer || 
        params.resource_type == ResourceType::TextureLinear)
    {
        if (params.resource_type == ResourceType::StructuredBuffer)
        {
            auto buffer_size = sizeof(float3) * params.element_count;

            // Test buffer for asking about its memory properties
            auto test_buffer = createBuffer(1, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            VkMemoryRequirements requirements;
            vkGetBufferMemoryRequirements(logical_device, test_buffer, &requirements);
            requirements.size = getAlignedSize(buffer_size, requirements.alignment);
            vkDestroyBuffer(logical_device, test_buffer, nullptr);

            // Allocate memory and bind it to buffers
            view.aux_memory = allocateMemory(requirements,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            );
            view.vertex_buffer = createBuffer(buffer_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            vkBindBufferMemory(logical_device, view.vertex_buffer, view.aux_memory, 0);

            float3 *data = nullptr;
            vkMapMemory(logical_device, view.aux_memory, 0, buffer_size, 0, (void**)&data);
            auto vk_extent = view.vk_extent;
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
            vkUnmapMemory(logical_device, view.aux_memory);

            deletors.pushFunction([=]{
                vkDestroyBuffer(logical_device, view.vertex_buffer, nullptr);
                vkFreeMemory(logical_device, view.aux_memory, nullptr);
            });
        }

        // Create view buffers
        VkDeviceSize memsize = params.element_size * params.element_count;
        auto usage = getUsageFlags(params.primitive_type, params.resource_type);
        VkExternalMemoryBufferCreateInfo extmem_info{};
        extmem_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        extmem_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        view.data_buffer = createBuffer(memsize, usage, &extmem_info);
        deletors.pushFunction([=]{
            vkDestroyBuffer(logical_device, view.data_buffer, nullptr);
        });
        vkGetBufferMemoryRequirements(logical_device, view.data_buffer, &bufreq);
    }
    if (params.resource_type == ResourceType::Texture ||
        params.resource_type == ResourceType::TextureLinear)
    {
        // Init texture memory
        auto img_type = getImageType(params.data_domain);
        VkExternalMemoryImageCreateInfo ext_info{};
        ext_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        ext_info.pNext = nullptr;
        ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        auto img_format = getVulkanFormat(params.texture_format);
        VkExtent3D img_extent = {params.extent.x, params.extent.y, params.extent.z};
        auto tiling = getImageTiling(params.resource_type);
        auto usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        view.image = createImage(img_type, img_format, img_extent, tiling, usage, &ext_info);
        view.vk_sampler = createSampler(VK_FILTER_NEAREST, true);
        deletors.pushFunction([=]{
            vkDestroyImage(logical_device, view.image, nullptr);
        });
        vkGetImageMemoryRequirements(logical_device, view.image, &imgreq);
    }

    if (params.resource_type == ResourceType::StructuredBuffer || 
        params.resource_type == ResourceType::UnstructuredBuffer)
    {
        memreq = bufreq;
    }
    else if (params.resource_type == ResourceType::Texture)
    {
        memreq = imgreq;
    }
    else if (params.resource_type == ResourceType::TextureLinear)
    {
        memreq.size = std::max(bufreq.size, imgreq.size);
        memreq.alignment = std::max(bufreq.size, imgreq.size);
        memreq.memoryTypeBits = bufreq.memoryTypeBits & imgreq.memoryTypeBits;
    }

    VkExportMemoryAllocateInfoKHR export_info{};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    export_info.pNext = nullptr;
    export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    auto properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    view.memory = allocateMemory(memreq, properties, &export_info);
    importCudaExternalMemory(view.cuda_extmem, view.memory, memreq.size);
    deletors.pushFunction([=]{
        validation::checkCuda(cudaDestroyExternalMemory(view.cuda_extmem));
        vkFreeMemory(logical_device, view.memory, nullptr);
    });

    if (params.resource_type == ResourceType::StructuredBuffer || 
        params.resource_type == ResourceType::UnstructuredBuffer || 
        params.resource_type == ResourceType::TextureLinear)
    {
        VkDeviceSize memsize = params.element_size * params.element_count;
        vkBindBufferMemory(logical_device, view.data_buffer, view.memory, 0);
        cudaExternalMemoryBufferDesc buffer_desc{};
        buffer_desc.offset = 0;
        buffer_desc.size   = memsize;
        buffer_desc.flags  = 0;
        validation::checkCuda(cudaExternalMemoryGetMappedBuffer(
            &view.cuda_ptr, view.cuda_extmem, &buffer_desc)
        );
    }
    if (params.resource_type == ResourceType::Texture ||
        params.resource_type == ResourceType::TextureLinear)
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
        auto usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        auto requirements = getMemoryRequiements(usage, {vert_size, ids_size});
        auto max_aligned_size = requirements.size;
        // The largest alignment requirement can be used for all
        requirements.size = max_aligned_size * 2;

        // Allocate memory and bind it to buffers
        view.aux_memory = allocateMemory(requirements,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        view.vertex_buffer = createBuffer(vert_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        vkBindBufferMemory(logical_device, view.vertex_buffer, view.aux_memory, 0);
        char *data = nullptr;
        vkMapMemory(logical_device, view.aux_memory, 0, vert_size, 0, (void**)&data);
        std::memcpy(data, vertices.data(), vert_size);
        vkUnmapMemory(logical_device, view.aux_memory);

        view.index_buffer = createBuffer(ids_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        vkBindBufferMemory(logical_device, view.index_buffer, view.aux_memory, max_aligned_size);
        data = nullptr;
        vkMapMemory(logical_device, view.aux_memory, max_aligned_size, ids_size, 0, (void**)&data);
        std::memcpy(data, indices.data(), ids_size);
        vkUnmapMemory(logical_device, view.aux_memory);

        deletors.pushFunction([=]{
            vkDestroyBuffer(logical_device, view.vertex_buffer, nullptr);
            vkDestroyBuffer(logical_device, view.index_buffer, nullptr);
            vkFreeMemory(logical_device, view.aux_memory, nullptr);
        });

        vkBindImageMemory(logical_device, view.image, view.memory, 0);

        auto view_type = getViewType(params.data_domain);
        auto info = vkinit::imageViewCreateInfo(view.image,
            view_type, view.vk_format, VK_IMAGE_ASPECT_COLOR_BIT
        );
        validation::checkVulkan(vkCreateImageView(logical_device, &info, nullptr, &view.vk_view));
        
        deletors.pushFunction([=]{
            vkDestroyImageView(logical_device, view.vk_view, nullptr);
        });

        // Init texture memory (TODO: Refactor)
        if (params.resource_type == ResourceType::TextureLinear)
        {
            transitionImageLayout(view.image,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            );  
        }
        else if (params.resource_type == ResourceType::Texture)
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

            validation::checkCuda(cudaExternalMemoryGetMappedMipmappedArray(
                &view.mipmap_array, view.cuda_extmem, &array_desc)
            );

            deletors.pushFunction([=]{
                validation::checkCuda(cudaFreeMipmappedArray(view.mipmap_array));
            });
        }
    }
}

void VulkanCudaDevice::updateTexture(CudaView& view)
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
        vkCmdCopyBufferToImage(cmd, view.data_buffer, view.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
        );
    });

    transitionImageLayout(view.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );
}

void VulkanCudaDevice::loadTexture(CudaView *view, void *img_data)
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
    VkDeviceSize memsize = params.element_size * params.element_count;
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
    immediateSubmit([=](VkCommandBuffer cmd)
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

void VulkanCudaDevice::importCudaExternalMemory(
    cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size)
{
    cudaExternalMemoryHandleDesc extmem_desc{};
    extmem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    extmem_desc.size = size;
    extmem_desc.handle.fd = (int)(uintptr_t)getMemoryHandle(
        vk_mem, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    );
    validation::checkCuda(cudaImportExternalMemory(&cuda_mem, &extmem_desc));
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
