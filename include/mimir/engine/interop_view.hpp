#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include <string> // std::string

#include <mimir/view_types.hpp>

namespace mimir
{

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
    uint3 extent = {1, 1, 1};
    DataType data_type;
    DataDomain data_domain;
    DomainType domain_type;
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
    // Cuda external memory handle, provided by the Cuda interop API
    cudaExternalMemory_t cuda_extmem = nullptr;    

    // Raw Cuda pointer which can be passed to the library user
    // for use in kernels, as per cudaMalloc
    void *cuda_ptr = nullptr;
    // Vulkan buffer handle
    VkBuffer data_buffer = VK_NULL_HANDLE;
    // Vulkan external device memory
    VkDeviceMemory memory = VK_NULL_HANDLE;

    // Image members
    cudaMipmappedArray_t mipmap_array = nullptr;
    VkDeviceMemory image_memory = nullptr;
    VkImage image        = VK_NULL_HANDLE;
    VkImageView vk_view  = VK_NULL_HANDLE;
    VkSampler vk_sampler = VK_NULL_HANDLE;
    VkFormat vk_format   = VK_FORMAT_UNDEFINED;
    VkExtent3D vk_extent = {0, 0, 0};

    // Auxiliary buffer for storing vertex and index buffers
    VkBuffer aux_buffer       = VK_NULL_HANDLE;
    // Auxiliary memory allocation for the above
    VkDeviceMemory aux_memory = VK_NULL_HANDLE;
    // Offset in bytes where index buffer starts inside aux_memory
    VkDeviceSize index_offset = 0;

    // Switches view visibility from visible to invisible and viceversa.
    // Does not modify view data in any way
    bool toggleVisibility()
    {
        params.options.visible = !params.options.visible;
        return params.options.visible;
    }
};

// Converts a InteropView texture type to its Vulkan equivalent
constexpr VkFormat getDataFormat(DataType type, uint channel_count)
{
    switch (type)
    {
        case DataType::Int: switch (channel_count)
        {
            case 1: return VK_FORMAT_R32_SINT;
            case 2: return VK_FORMAT_R32G32_SINT;
            case 3: return VK_FORMAT_R32G32B32_SINT;
            case 4: return VK_FORMAT_R32G32B32A32_SINT;
            default: return VK_FORMAT_UNDEFINED;
        }
        case DataType::Char: switch (channel_count)
        {
            case 1: return VK_FORMAT_R8_SRGB;
            case 2: return VK_FORMAT_R8G8_SRGB;
            case 3: return VK_FORMAT_R8G8B8_SRGB;
            case 4: return VK_FORMAT_R8G8B8A8_SRGB;
            default: return VK_FORMAT_UNDEFINED;
        }
        case DataType::Float: switch (channel_count)
        {
            case 1: return VK_FORMAT_R32_SFLOAT;
            case 2: return VK_FORMAT_R32G32_SFLOAT;
            case 3: return VK_FORMAT_R32G32B32_SFLOAT;
            case 4: return VK_FORMAT_R32G32B32A32_SFLOAT;
            default: return VK_FORMAT_UNDEFINED;
        }
        case DataType::Double: switch (channel_count)
        {
            case 1: return VK_FORMAT_R64_SFLOAT;
            case 2: return VK_FORMAT_R64G64_SFLOAT;
            case 3: return VK_FORMAT_R64G64B64_SFLOAT;
            case 4: return VK_FORMAT_R64G64B64A64_SFLOAT;
            default: return VK_FORMAT_UNDEFINED;
        }        
        default: return VK_FORMAT_UNDEFINED;
    }
}

union DataSize
{
    int x;
    int2 xy;
    int3 xyz; 
};

enum class DataLayout { Layout1D, Layout2D, Layout3D };

constexpr size_t getElementCount(DataSize size, DataLayout layout)
{
    switch (layout)
    {
        case DataLayout::Layout1D: return size.x;
        case DataLayout::Layout2D: return size.xy.x * size.xy.y;
        case DataLayout::Layout3D: return size.xyz.x * size.xyz.y * size.xyz.z;
        default: return 0;
    }
};

struct MemoryParams
{
    DataLayout layout = DataLayout::Layout1D;
    DataSize element_count;
    uint channel_count = 1;
    DataType data_type;
    ResourceType resource_type;  
};

struct InteropMemory
{
    // View parameters
    MemoryParams params;

    // Cuda external memory handle, provided by the Cuda interop API
    cudaExternalMemory_t cuda_extmem = nullptr;

    // Raw Cuda pointer which can be passed to the library user
    // for use in kernels, as per cudaMalloc
    void *cuda_ptr = nullptr;
    // Vulkan buffer handle
    VkBuffer data_buffer = VK_NULL_HANDLE;
    // Vulkan external device memory
    VkDeviceMemory memory = VK_NULL_HANDLE;

    // Image members
    cudaMipmappedArray_t mipmap_array = nullptr;
    VkDeviceMemory image_memory = nullptr;
    VkImage image        = VK_NULL_HANDLE;
    VkImageView vk_view  = VK_NULL_HANDLE;
    VkSampler vk_sampler = VK_NULL_HANDLE;
    VkFormat vk_format   = VK_FORMAT_UNDEFINED;
    VkExtent3D vk_extent = {0, 0, 0};  
};

enum class AttributeType { Position, Color, Size };
enum class ViewType { Markers, Edges, Voxels, Image };

constexpr char* getDataLayout(DataLayout l)
{
    switch (l)
    {
        case DataLayout::Layout1D: return "1D";
        case DataLayout::Layout2D: return "2D";
        case DataLayout::Layout3D: return "3D";
        default: return "unknown";
    }
}

constexpr char* getAttributeType(AttributeType type)
{
    switch (type)
    {
#define STR(r) case AttributeType::r: return #r
        STR(Position);
        STR(Color);
        STR(Size);
#undef STR
        default: return "unknown";
    }
}

constexpr char* getViewType(ViewType type)
{
    switch (type)
    {
#define STR(r) case ViewType::r: return #r
        STR(Markers);
        STR(Edges);
        STR(Voxels);
        STR(Image);
#undef STR
        default: return "unknown";
    }
}

struct ViewAttribute
{
    InteropMemory memory;
    AttributeType type;
};

struct ViewOptions2
{
    // Customizable name for the view
    std::string name;
    // Flag indicating if this view should be displayed or not
    bool visible = true;
    // Default primitive color if no per-instance color is set
    float4 default_color{0.f,0.f,0.f,1.f};
    // Default primitive size if no per-instance size is set
    float default_size = 10.f;
    // External alternate shaders for use in this view
    std::vector<ShaderInfo> external_shaders;
    // For specializing slang shaders associated to this view
    std::vector<std::string> specializations;        
};

struct ViewParams2
{
    size_t element_count = 0;
    uint3 extent = {1, 1, 1};
    DataDomain data_domain;
    DomainType domain_type;
    ViewType view_type;
    ViewOptions2 options;
    std::vector<ViewAttribute> attributes;
};

struct InteropView2
{
    // View parameters
    ViewParams2 params;

    // Rendering pipeline associated to this view
    VkPipeline pipeline       = VK_NULL_HANDLE;
    // Auxiliary buffer for storing vertex and index buffers
    VkBuffer aux_buffer       = VK_NULL_HANDLE;
    // Auxiliary memory allocation for the above
    VkDeviceMemory aux_memory = VK_NULL_HANDLE;
    // Offset in bytes where index buffer starts inside aux_memory
    VkDeviceSize index_offset = 0;

    // Switches view visibility from visible to invisible and viceversa.
    // Does not modify view data in any way
    bool toggleVisibility()
    {
        params.options.visible = !params.options.visible;
        return params.options.visible;
    }
};

} // namespace mimir