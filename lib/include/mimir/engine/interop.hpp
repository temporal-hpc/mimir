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

cudaMipmappedArray_t createMipmapArray(cudaExternalMemory_t cuda_extmem,
    int4 component_size, int3 extent, unsigned level_count
);
cudaExternalMemory_t importCudaExternalMemory(
    VkDeviceMemory vk_mem, VkDeviceSize size, VkDevice device
);
cudaExternalSemaphore_t importCudaExternalSemaphore(
    VkSemaphore vk_semaphore, VkDevice device
);

} // namespace mimir::interop