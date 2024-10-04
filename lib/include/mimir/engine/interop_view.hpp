#pragma once

//#include "vk_mem_alloc.h"

#include <map> // std::map
#include <memory> // std::shared_ptr
#include <string> // std::string
#include <vector> // std::vector

#include <mimir/engine/view_types.hpp>

namespace mimir
{

enum class PresentMode { Immediate, TripleBuffering, VSync };

struct ShaderInfo
{
    std::string filepath;
    VkShaderStageFlagBits stage = VK_SHADER_STAGE_ALL_GRAPHICS;
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
    int scenario_index = 0;
    // For moving through the different slices in a 3D texture
    float depth = 0.1f;
    // External alternate shaders for use in this view
    std::vector<ShaderInfo> external_shaders;
    // For specializing slang shaders associated to this view
    std::vector<std::string> specializations;
    int custom_val = 0;
};

struct Allocation
{
    // Allocation memory size in bytes
    size_t size                      = 0;
    // Vulkan external device memory handle
    VkDeviceMemory vk_mem            = VK_NULL_HANDLE;
    // Cuda external memory handle, provided by the Cuda interop API
    cudaExternalMemory_t cuda_extmem = nullptr;

    // VMA object representing the underlying memory
    // VmaAllocation allocation      = nullptr;
};

struct AttributeParams
{
    // Interop memory handle
    std::shared_ptr<Allocation> allocation = nullptr;
    // Type of variables stored per element
    DataFormat format = {};
    // Offset to start of memory handle
    VkDeviceSize offset = 0;
};

using AttributeDict = std::map<AttributeType, AttributeParams>;

struct ViewParams
{
    size_t element_count = 0;
    uint3 extent = {1, 1, 1};
    DomainType data_domain;
    ViewType view_type;
    ViewOptions options;
    AttributeDict attributes;
    AttributeParams indexing;
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> sizes;
};

// Container for all vertex buffer objects associated to a Mimir view object.
struct BufferArray
{
    // Number of vertex buffers in the view.
    // The sizes of the handles and offsets arrays equals this value.
    uint32_t count = 0;
    // Set of vertex buffer objects associated to the view.
    std::vector<VkBuffer> handles;
    // Start region or each buffer object in the set.
    std::vector<VkDeviceSize> offsets;
};

// If the contained buffer handle is not null, the associated
// view will bind said handle as an index buffer when drawing.
struct IndexBuffer
{
    // Handle to index buffer object.
    VkBuffer handle  = VK_NULL_HANDLE;
    // Specify size of each index datum.
    VkIndexType type = VK_INDEX_TYPE_UINT32;
};

struct ImageData
{
    VkImage handle       = VK_NULL_HANDLE;
    VkImageView img_view = VK_NULL_HANDLE;
    VkSampler sampler    = VK_NULL_HANDLE;
    VkFormat format      = VK_FORMAT_UNDEFINED;
    VkExtent3D extent    = {0, 0, 0};
};

struct ViewResources
{
    BufferArray vbo;
    IndexBuffer ibo;
    BufferArray ubo;
    ImageData image;
};

struct InteropView
{
    // View parameters
    ViewParams params;

    // Container for Vulkan graphics resources.
    ViewResources resources;
    // Rendering pipeline associated to this view.
    VkPipeline pipeline = VK_NULL_HANDLE;

    // Switches view state between visible and invisible; does not modify underlying data.
    bool toggleVisibility()
    {
        params.options.visible = !params.options.visible;
        return params.options.visible;
    }
};

} // namespace mimir