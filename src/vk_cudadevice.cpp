#include "cudaview/engine/vk_cudadevice.hpp"

#include <cuda_runtime.h>

#include <cstring> // std::memcpy

#include "cudaview/vk_types.hpp"
#include "internal/vk_initializers.hpp"
#include "internal/color.hpp"
#include "internal/utils.hpp"
#include "internal/validation.hpp"

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
