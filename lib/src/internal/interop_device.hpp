#pragma once

#include <cuda_runtime_api.h>

#include <mimir/interop_view.hpp>

namespace mimir
{

// Class for encapsulating Vulkan device functions with Cuda interop
// Inherited from VulkanDevice to encapsulate Cuda-related code
struct InteropDevice
{
    void initMemoryBuffer(InteropMemory &interop);
    void initMemoryImage(InteropMemory &interop);
    void initMemoryImageLinear(InteropMemory &interop);
    void initViewBuffer(InteropViewOld& view);
    void initViewImage(InteropViewOld& view);

    void loadTexture(InteropMemory *interop, void *data);
    void copyBufferToTexture(VkBuffer buffer, VkImage image, VkExtent3D extent);
    void updateLinearTexture(InteropMemory &interop);
    VkImage createImage2(MemoryParams params);
    uint32_t getMaxImageDimension(DataLayout layout);
};

} // namespace mimir