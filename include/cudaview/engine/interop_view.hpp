#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include <string> // std::string

enum class PresentOptions { Immediate, TripleBuffering, VSync };

// Specifies the number of spatial dimensions of the view
enum class DataDomain    { Domain2D, Domain3D };
// Specifies the data layout
enum class ResourceType  { UnstructuredBuffer, StructuredBuffer, Texture, TextureLinear };
// Specifies the type of primitive that will be visualized 
enum class PrimitiveType { Points, Edges, Voxels };
// Specifies the datatype stored in the texture corresponding to a view
enum class DataType { Int1, Int2, Int3, Int4, Float1, Float2, Float3, Float4, Char1, Char2, Char3, Char4 };

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
    uint3 extent = {1, 1, 1};
    DataDomain data_domain;
    ResourceType resource_type;
    PrimitiveType primitive_type;
    DataType data_type;
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

    bool toggleVisibility()
    {
        params.options.visible = !params.options.visible;
        return params.options.visible;
    }
};

constexpr size_t getDataSize(DataType t)
{
    switch (t)
    {

        case DataType::Int1: return sizeof(int);
        case DataType::Int2: return sizeof(int2);
        case DataType::Int3: return sizeof(int3);
        case DataType::Int4: return sizeof(int4);
        case DataType::Float1: return sizeof(float);
        case DataType::Float2: return sizeof(float2);
        case DataType::Float3: return sizeof(float3);
        case DataType::Float4: return sizeof(float4);
        case DataType::Char1: return sizeof(char);
        case DataType::Char2: return sizeof(char2);
        case DataType::Char3: return sizeof(char3);
        case DataType::Char4: return sizeof(char4);
        default: return 0;
    }
}