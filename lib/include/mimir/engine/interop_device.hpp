#pragma once

#include <cuda_runtime_api.h>

#include <mimir/engine/interop_view.hpp>
#include <mimir/engine/vulkan_device.hpp>

namespace mimir
{

struct InteropBarrier
{
    uint64_t timeline_value                = 0;
    VkSemaphore vk_semaphore               = VK_NULL_HANDLE;
    cudaExternalSemaphore_t cuda_semaphore = nullptr;
    cudaStream_t cuda_stream               = 0;
};

// Class for encapsulating Vulkan device functions with Cuda interop
// Inherited from VulkanDevice to encapsulate Cuda-related code
struct InteropDevice : public VulkanDevice
{
    // Use the constructor from the base VulkanDevice class
    using VulkanDevice::VulkanDevice;

    void initMemoryBuffer(InteropMemory &interop);
    void initMemoryImage(InteropMemory &interop);
    void initMemoryImageLinear(InteropMemory &interop);
    void initViewBuffer(InteropViewOld& view);
    void initViewImage(InteropViewOld& view);

    void loadTexture(InteropMemory *interop, void *data);
    void copyBufferToTexture(VkBuffer buffer, VkImage image, VkExtent3D extent);
    void updateLinearTexture(InteropMemory &interop);
    VkImage createImage(MemoryParams params);
    uint32_t getMaxImageDimension(DataLayout layout);

    InteropBarrier createInteropBarrier();
};

} // namespace mimir