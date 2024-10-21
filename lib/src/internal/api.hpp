#pragma once

#include <mimir/view.hpp>

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

namespace mimir
{

constexpr uint32_t max_attr_count = 10;

struct Allocation
{
    // Allocation memory size in bytes
    VkDeviceSize size;
    // Vulkan external device memory handle
    VkDeviceMemory vk_mem;
    // Cuda external memory handle, provided by the Cuda interop API
    cudaExternalMemory_t cuda_extmem;
};

struct ViewDetails
{
    // Rendering pipeline associated to this view.
    VkPipeline pipeline;
    // Number of vertices or indices to draw; derived from element count in view description.
    uint32_t draw_count;
    // Number of vertex buffers in the view.
    uint32_t vb_count;
    // Array of vertex buffer objects associated to the view.
    VkBuffer vbo[max_attr_count];
    // Start region for each buffer object in the view.
    VkDeviceSize offsets[max_attr_count];
    // Set to true at view creation when using an index buffer for drawing; false otherwise.
    bool use_ibo;
    // Index buffer used when is_indexed is true. Uninitialized if view uses direct source mapping.
    VkBuffer ibo;
    // Index type used when is_indexed is true; undefined otherwise.
    VkIndexType index_type;
    // Copy of description used to create this view.
    ViewDescription desc;
};

constexpr char* getViewType(ViewType type)
{
    switch (type)
    {
#define STR(r) case ViewType::r: return (char*)#r
        STR(Markers);
        STR(Edges);
        STR(Image);
        STR(Boxes);
#undef STR
        default: return (char*)"unknown";
    }
}

constexpr char* getDomainType(DomainType d)
{
    switch (d)
    {
        case DomainType::Domain2D: return (char*)"2D";
        case DomainType::Domain3D: return (char*)"3D";
        default: return (char*)"unknown";
    }
}

constexpr char* getAttributeType(AttributeType type)
{
    switch (type)
    {
#define STR(r) case AttributeType::r: return (char*)#r
        STR(Position);
        STR(Color);
        STR(Size);
#undef STR
        default: return (char*)"unknown";
    }
}

constexpr char* getDataType(FormatDescription desc)
{
    switch (desc.kind)
    {
        case FormatKind::Float: switch (desc.size)
        {
            case 4: return (char*)"Float";
            case 8: return (char*)"Double";
            case 2: return (char*)"Half";
            default: return (char*)"unknown";
        }
        // TODO: Output 'u' prefix when unsigned
        case FormatKind::Signed: case FormatKind::Unsigned: switch (desc.size)
        {
            case 4: return (char*)"Int";
            case 8: return (char*)"Long";
            case 2: return (char*)"Short";
            case 1: return (char*)"Char";
            default: return (char*)"unknown";
        }
        default: return (char*)"unknown";
    }
}

} // namespace mimir