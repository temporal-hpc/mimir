#include <mimir/engine/interop.hpp>

#include <cuda_runtime.h> // make_cudaExtent

#include "internal/resources.hpp"
#include "internal/validation.hpp"

namespace mimir::interop
{

Barrier Barrier::make(VkDevice device)
{
    Barrier barrier{
        .timeline_value = 0,
        .vk_semaphore   = VK_NULL_HANDLE,
        .cuda_semaphore = nullptr,
        .cuda_stream    = 0,
    };

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
    barrier.vk_semaphore   = createSemaphore(device, &export_info);
    barrier.cuda_semaphore = importCudaExternalSemaphore(barrier.vk_semaphore, device);
    return barrier;
}

cudaMipmappedArray_t createMipmapArray(cudaExternalMemory_t cuda_extmem,
    int4 component_size, int3 extent, unsigned level_count)
{
    cudaChannelFormatDesc format_desc{
        .x = component_size.x,
        .y = component_size.z,
        .z = component_size.y,
        .w = component_size.w,
        .f = cudaChannelFormatKindUnsigned
    };
    cudaExternalMemoryMipmappedArrayDesc array_desc{
        .offset     = 0,
        .formatDesc = format_desc,
        .extent     = make_cudaExtent(extent.x, extent.y, extent.z),
        .flags      = 0,
        .numLevels  = level_count,
    };

    cudaMipmappedArray_t mipmap_array;
    validation::checkCuda(cudaExternalMemoryGetMappedMipmappedArray(
        &mipmap_array, cuda_extmem, &array_desc)
    );
    return mipmap_array;
}

cudaExternalMemory_t importCudaExternalMemory(
    VkDeviceMemory vk_mem, VkDeviceSize size, VkDevice device)
{
    // Get external memory handle function
    VkMemoryGetFdInfoKHR fd_info{
        .sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .pNext      = nullptr,
        .memory     = vk_mem,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    auto fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(
        device, "vkGetMemoryFdKHR"
    );
    if (!fpGetMemoryFdKHR)
    {
        spdlog::error("Failed to retrieve vkGetMemoryFdKHR function handle!");
    }
    // Get external memory handle
    int fd = -1;
    validation::checkVulkan(fpGetMemoryFdKHR(device, &fd_info, &fd));

    cudaExternalMemoryHandleDesc extmem_desc{};
    extmem_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    extmem_desc.size      = size;
    extmem_desc.handle.fd = fd;
    cudaExternalMemory_t cuda_mem;
    validation::checkCuda(cudaImportExternalMemory(&cuda_mem, &extmem_desc));
    return cuda_mem;
}

cudaExternalSemaphore_t importCudaExternalSemaphore(
    VkSemaphore vk_semaphore, VkDevice device)
{
    // Get external semaphore handle function
    auto fpGetSemaphore = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
        device, "vkGetSemaphoreFdKHR"
    );
    if (!fpGetSemaphore)
    {
        spdlog::error("Failed to retrieve vkGetSemaphoreFdKHR function handle!");
    }
    VkSemaphoreGetFdInfoKHR fd_info{
        .sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
        .pNext      = nullptr,
        .semaphore  = vk_semaphore,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    // Get external semaphore handle
    int fd = -1;
    validation::checkVulkan(fpGetSemaphore(device, &fd_info, &fd));

    cudaExternalSemaphoreHandleDesc desc{};
    desc.type      = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    desc.handle.fd = fd;
    desc.flags     = 0;

    cudaExternalSemaphore_t cuda_semaphore;
    validation::checkCuda(cudaImportExternalSemaphore(&cuda_semaphore, &desc));
    return cuda_semaphore;
}

} // namespace mimir::interop