#pragma once

#include "cudaview/engine/vk_device.hpp"

#include <cuda_runtime_api.h>

#include <cudaview/engine/cudaview.hpp>
#include <cudaview/vk_types.hpp>

struct FrameBarrier
{
    VkSemaphore present_semaphore = VK_NULL_HANDLE;
    VkSemaphore render_semaphore  = VK_NULL_HANDLE;
    VkFence render_fence = VK_NULL_HANDLE;
};

struct InteropBarrier
{
    VkSemaphore vk_semaphore = VK_NULL_HANDLE;
    cudaExternalSemaphore_t cuda_semaphore = nullptr;
};

// Class for encapsulating Vulkan device functions with Cuda interop 
// Inherited from VulkanDevice to encapsulate Cuda-related code
struct VulkanCudaDevice : public VulkanDevice
{
    // Use the constructor from the base VulkanDevice class
    using VulkanDevice::VulkanDevice;

    InteropBarrier createInteropBarrier();
    void importCudaExternalMemory(cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size);
    void *getMemoryHandle(VkDeviceMemory memory,
        VkExternalMemoryHandleTypeFlagBits handle_type
    );
    void *getSemaphoreHandle(VkSemaphore semaphore,
        VkExternalSemaphoreHandleTypeFlagBits handle_type
    );

    // View functions
    void initView(CudaView& view);
    void updateTexture(CudaView& view);
    void loadTexture(CudaView *view, void *data);
    void createUniformBuffers(CudaView& view, uint32_t img_count);
    void updateUniformBuffers(CudaView& view, uint32_t image_idx,
        ModelViewProjection mvp, PrimitiveParams options, SceneParams scene
    );    
};
