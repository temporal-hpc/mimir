#pragma once

#include <map> // std::map
#include <string> // std::string
#include <vector> // std::vector

#include <mimir/view_types.hpp>

namespace mimir
{

enum class PresentOptions { Immediate, TripleBuffering, VSync };

struct ShaderInfo
{
    std::string filepath;
    VkShaderStageFlagBits stage = VK_SHADER_STAGE_ALL_GRAPHICS;
};

constexpr VkIndexType getIndexType(ComponentType type)
{
    switch (type)
    {
        case ComponentType::Int: return VK_INDEX_TYPE_UINT32;
        // TODO: Add VK_INDEX_TYPE_UINT8_EXT for char
        // and VK_INDEX_TYPE_NONE_KHR for default
        default: return VK_INDEX_TYPE_UINT16;
    }
}

struct MemoryParams
{
    DataLayout layout = DataLayout::Layout1D;
    uint3 element_count = {1, 1, 1};
    uint channel_count = 1;
    ComponentType component_type;
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
    // TODO: Move sampler creation to engine
    // But the sampler field should still remain here
    VkSampler vk_sampler = VK_NULL_HANDLE;
    VkFormat vk_format   = VK_FORMAT_UNDEFINED;
    VkExtent3D vk_extent = {0, 0, 0};
};

struct ViewOptions
{
    // Customizable name for the view
    std::string name;
    // Flag indicating if this view should be displayed or not
    bool visible = true;
    // Default primitive color if no per-instance color is set
    float4 default_color{0.f,0.f,0.f,1.f};
    // Default primitive size if no per-instance size is set
    float default_size = 10.f;
    // Index of instance inside view
    int instance_index = 0;
    // For moving through the different slices in a 3D texture
    float depth = 0.1f;
    // External alternate shaders for use in this view
    std::vector<ShaderInfo> external_shaders;
    // For specializing slang shaders associated to this view
    std::vector<std::string> specializations;
    int custom_val = 0;
};

using AttributeDict = std::map<AttributeType, InteropMemory>;

struct ViewParams
{
    size_t element_count = 0;
    int instance_count = 1;
    uint3 extent = {1, 1, 1};
    DataDomain data_domain;
    DomainType domain_type;
    ViewType view_type;
    ViewOptions options;
    AttributeDict attributes;
};

struct InteropView
{
    // View parameters
    ViewParams params;

    // Rendering pipeline associated to this view
    VkPipeline pipeline       = VK_NULL_HANDLE;
    // Auxiliary buffer for storing vertex and index buffers
    VkBuffer aux_buffer       = VK_NULL_HANDLE;
    // Auxiliary memory allocation for the above
    VkDeviceMemory aux_memory = VK_NULL_HANDLE;
    // Offset in bytes where index buffer starts inside aux_memory
    VkDeviceSize index_offset = 0;

    std::vector<VkBuffer> vert_buffers;
    std::vector<VkDeviceSize> buffer_offsets;
    VkBuffer idx_buffer = VK_NULL_HANDLE;
    VkIndexType idx_type;

    // Switches view visibility from visible to invisible and viceversa.
    // Does not modify view data in any way
    bool toggleVisibility()
    {
        params.options.visible = !params.options.visible;
        return params.options.visible;
    }
};

} // namespace mimir