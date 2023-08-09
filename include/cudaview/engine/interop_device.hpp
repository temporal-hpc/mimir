#pragma once

#include <cuda_runtime_api.h>

#include <cudaview/engine/vulkan_device.hpp>
#include <cudaview/engine/interop_view.hpp>
#include <cudaview/vk_types.hpp>

struct FrameBarrier
{
    VkSemaphore present_semaphore = VK_NULL_HANDLE;
    VkSemaphore render_semaphore  = VK_NULL_HANDLE;
    VkFence render_fence = VK_NULL_HANDLE;
};

struct InteropBarrier
{
    cudaStream_t cuda_stream = 0;
    VkSemaphore vk_semaphore = VK_NULL_HANDLE;
    cudaExternalSemaphore_t cuda_semaphore = nullptr;
};

// Class for encapsulating Vulkan device functions with Cuda interop 
// Inherited from VulkanDevice to encapsulate Cuda-related code
struct InteropDevice : public VulkanDevice
{
    // Use the constructor from the base VulkanDevice class
    using VulkanDevice::VulkanDevice;

    void importCudaExternalMemory(cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size);
    void *getMemoryHandle(VkDeviceMemory memory,
        VkExternalMemoryHandleTypeFlagBits handle_type
    );
    void *getSemaphoreHandle(VkSemaphore semaphore,
        VkExternalSemaphoreHandleTypeFlagBits handle_type
    );
    InteropBarrier createInteropBarrier();

    // View functions
    void initView(InteropView& view);
    void updateTexture(InteropView& view);
    void loadTexture(InteropView *view, void *data);
};
