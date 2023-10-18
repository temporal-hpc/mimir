#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include <string> // std::string

#include <cudaview/view_types.hpp>

enum class PresentOptions { Immediate, TripleBuffering, VSync };

struct ShaderInfo
{
    std::string filepath;
    VkShaderStageFlagBits stage = VK_SHADER_STAGE_ALL_GRAPHICS;
};

// Holds customization options for the view it is associated to, for example:
// Point color, point size, edge color, etc.
// In the future, it should only have the fields the view actually supports
struct ViewOptions
{
    // Customizable name for the view
    std::string name;
    // Flag indicating if this view should be displayed or not
    bool visible = true;
    // Default primitive color if no per-instance color is set
    float4 color{0.f,0.f,0.f,1.f};
    // Default primitive size if no per-instance size is set
    float size = 10.f;
    // External alternate shaders for use in this view
    std::vector<ShaderInfo> external_shaders;
    // For specializing slang shaders associated to this view
    std::vector<std::string> specializations;
    // For moving through the different slices in a 3D texture
    float depth = 0.1f;
};

struct ViewParams
{
    cudaStream_t cuda_stream = 0;
    size_t element_count = 0;
    uint channel_count = 1;
    DataType data_type;
    uint3 extent = {1, 1, 1};
    DataDomain data_domain;
    ResourceType resource_type;
    ElementType element_type;
    ViewOptions options;
};

struct InteropView
{
    // View parameters
    ViewParams params;

    // Rendering pipeline associated to this view
    VkPipeline pipeline = VK_NULL_HANDLE;

    // Auxiliary buffer for storing vertex and index buffers
    VkBuffer aux_buffer       = VK_NULL_HANDLE;
    // Auxiliary memory allocation for the above
    VkDeviceMemory aux_memory = VK_NULL_HANDLE;
    // Offset in bytes where index buffer starts inside aux_memory
    VkDeviceSize index_offset = 0;

    // Raw Cuda pointer which can be passed to the library user
    // for use in kernels, as per cudaMalloc
    void *cuda_ptr = nullptr;
    // Vulkan buffer handle
    VkBuffer data_buffer = VK_NULL_HANDLE;
    // Vulkan external device memory
    VkDeviceMemory memory = VK_NULL_HANDLE;  
    // Cuda external memory handle, provided by the Cuda interop API
    cudaExternalMemory_t cuda_extmem = nullptr;

    // Image members
    cudaMipmappedArray_t mipmap_array = nullptr;
    VkImage image        = VK_NULL_HANDLE;
    VkImageView vk_view  = VK_NULL_HANDLE;
    VkSampler vk_sampler = VK_NULL_HANDLE;
    VkFormat vk_format   = VK_FORMAT_UNDEFINED;
    VkExtent3D vk_extent = {0, 0, 0};

    // Switches view visibility from visible to invisible and viceversa.
    // Does not modify view data in any way
    bool toggleVisibility()
    {
        params.options.visible = !params.options.visible;
        return params.options.visible;
    }
};
