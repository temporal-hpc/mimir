#include "internal/interop.hpp"

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

cudaExternalMemory_t importCudaExternalMemory(
    VkDeviceMemory vk_mem, VkDeviceSize size, VkDevice device)
{
    cudaExternalMemory_t cuda_mem = nullptr;
    // Get external memory handle function
    VkMemoryGetFdInfoKHR fd_info{
        .sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .pNext      = nullptr,
        .memory     = vk_mem,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    auto vkGetMemoryFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFd)
    {
        spdlog::error("Failed to retrieve vkGetMemoryFdKHR function handle");
        return cuda_mem;
    }
    // Get external memory handle
    int fd = -1;
    validation::checkVulkan(vkGetMemoryFd(device, &fd_info, &fd));

    cudaExternalMemoryHandleDesc extmem_desc{
        .type   = cudaExternalMemoryHandleTypeOpaqueFd,
        .handle = {.fd = fd},
        .size   = size,
        .flags  = 0
    };
    validation::checkCuda(cudaImportExternalMemory(&cuda_mem, &extmem_desc));
    return cuda_mem;
}

cudaExternalSemaphore_t importCudaExternalSemaphore(
    VkSemaphore vk_semaphore, VkDevice device)
{
    cudaExternalSemaphore_t cuda_semaphore = nullptr;
    // Get external semaphore handle function
    auto vkGetSemaphoreFd = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
        device, "vkGetSemaphoreFdKHR"
    );
    if (!vkGetSemaphoreFd)
    {
        spdlog::error("Failed to retrieve vkGetSemaphoreFdKHR function handle");
        return cuda_semaphore;
    }
    VkSemaphoreGetFdInfoKHR fd_info{
        .sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
        .pNext      = nullptr,
        .semaphore  = vk_semaphore,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    // Get external semaphore handle
    int fd = -1;
    validation::checkVulkan(vkGetSemaphoreFd(device, &fd_info, &fd));

    cudaExternalSemaphoreHandleDesc desc{
        .type   = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd,
        .handle = { .fd = fd },
        .flags  = 0,
    };

    validation::checkCuda(cudaImportExternalSemaphore(&cuda_semaphore, &desc));
    return cuda_semaphore;
}

} // namespace mimir::interop