#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime_api.h>

namespace mimir::interop
{

struct Barrier
{
    uint64_t timeline_value;
    VkSemaphore vk_semaphore;
    cudaExternalSemaphore_t cuda_semaphore;
    cudaStream_t cuda_stream;

    static Barrier make(VkDevice device);
};

cudaExternalMemory_t importCudaExternalMemory(
    VkDeviceMemory vk_mem, VkDeviceSize size, VkDevice device
);
cudaExternalSemaphore_t importCudaExternalSemaphore(
    VkSemaphore vk_semaphore, VkDevice device
);

} // namespace mimir::interop