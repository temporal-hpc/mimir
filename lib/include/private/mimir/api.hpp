#pragma once

#include <mimir/view.hpp>

#include <glm/mat4x4.hpp>

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

namespace mimir
{

// Limit for number of attributes defined for a view
constexpr uint32_t max_attr_count = 10;
// Number of AttributeType enumerators (Position, Color, Size, Rotation, Texcoord); indexes attr_vbo.
constexpr uint32_t attr_type_count = 5;

struct LinearAlloc
{
    // Allocation memory size in bytes.
    VkDeviceSize size;
    // Vulkan external device memory handle.
    VkDeviceMemory vk_mem;
    // Cuda external memory handle, provided by the Cuda interop API.
    cudaExternalMemory_t cuda_extmem;
    // Cuda-mapped device pointer aliasing this buffer's memory (the same pointer handed back to the
    // caller of allocLinear). Persisted so consumers -- e.g. the LOD CUDA reduction -- can read the
    // interop positions as a native CUDA device pointer without re-mapping. nullptr until set.
    void *cuda_ptr = nullptr;
};

struct OpaqueAlloc
{
    // Allocation memory size in bytes.
    VkDeviceSize size;
    // Vulkan external device memory handle.
    VkDeviceMemory vk_mem;
    // Cuda external memory handle, provided by the Cuda interop API.
    cudaExternalMemory_t cuda_extmem;
    // Format description for texels.
    FormatDescription format = {};
    // Amount of texels in (width, height, depth) format.
    Layout extent = {};
    // Number of mipmap levels stored in the texture.
    unsigned int levels = 1;
};

struct Texture
{
    // Texture image handle.
    VkImage image;
    // Image view handle.
    VkImageView img_view;
    // Texture image sampler.
    // TODO: Move sampler creation to engine and reference the created sampler here
    VkSampler sampler;
    // Texture image format.
    VkFormat format;
    // Texture image dimensions.
    VkExtent3D extent;
    // Copy-path fields, set only when the interop buffer CANNOT be aliased directly to the image
    // (the device's LINEAR row pitch would differ from the buffer's tight packing and shear). Then
    // `image` is a device-local image, `copy_src` is a buffer over the shared interop memory, and
    // `image` is refreshed from `copy_src` each frame via vkCmdCopyBufferToImage (no host round-trip).
    // Both are VK_NULL_HANDLE on the zero-copy alias path.
    VkBuffer       copy_src = VK_NULL_HANDLE;
    VkDeviceMemory own_mem  = VK_NULL_HANDLE;
};

struct View
{
    // Rendering pipeline associated to this view.
    VkPipeline pipeline;
    // Number of vertices or indices to draw; derived from element count in view description.
    uint32_t draw_count;
    // Instances to draw (default 1). >1 for instanced marker meshes: draw_count is the template
    // icosphere's index count and instance_count is the particle count (SphereMesh render mode).
    uint32_t instance_count = 1;
    // True element (particle/vertex) total — 64-bit source of truth for the chunk loops.
    // draw_count/instance_count stay uint32 (per-chunk or the mesh index count).
    uint64_t element_count = 0;
    // Number of vertex buffers in the view.
    uint32_t vb_count;
    // Array of vertex buffer objects associated to the view.
    VkBuffer vbo[max_attr_count];
    // Start region for each buffer object in the view.
    VkDeviceSize offsets[max_attr_count];
    // Vertex-buffer slot (index into vbo[]) each attribute type landed in, or -1 when the attribute
    // is absent or was not mapped as a vertex buffer (indexed attributes go to `storage` instead).
    // Lets a non-raster consumer find an attribute's buffer without assuming a binding order --
    // the path tracer uses it to reach the Color attribute for per-primitive albedo.
    int attr_vbo[attr_type_count] = {-1, -1, -1, -1, -1};
    // Per-binding element stride (bytes) and input rate, so the chunked draw can advance each binding
    // by chunk_start * stride for the bindings whose rate matches the chunked dimension.
    VkDeviceSize      vbo_stride[max_attr_count] = {0};
    VkVertexInputRate vbo_rate[max_attr_count]   = {VK_VERTEX_INPUT_RATE_VERTEX};
    // Set to true at view creation when using an index buffer for drawing; false otherwise.
    bool use_ibo;
    // Index buffer used when is_indexed is true. Uninitialized if view uses direct source mapping.
    VkBuffer ibo;
    // Index type used when is_indexed is true; undefined otherwise.
    VkIndexType index_type;
    // Number of attached textures in the view.
    uint32_t tex_count;
    // Array of stored textures.
    Texture textures[max_attr_count];
    // Number of storage buffers in the view.
    uint32_t ssbo_count;
    // Storage buffer array.
    VkBuffer storage[max_attr_count];
    // Model matrices
    glm::mat4x4 translation, rotation, scale;

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

// Names for the two draw axes (ViewerOptions::shading / ::geometry), for log messages.
constexpr char* getShading(Shading s)
{
    switch (s)
    {
#define STR(r) case Shading::r: return (char*)#r
        STR(Unlit);
        STR(Phong);
        STR(PathTraced);
#undef STR
        default: return (char*)"unknown";
    }
}

constexpr char* getGeometry(Geometry g)
{
    switch (g)
    {
#define STR(r) case Geometry::r: return (char*)#r
        STR(Sprite);
        STR(Impostor);
        STR(Mesh);
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
        STR(Rotation);
        STR(Texcoord);
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

constexpr char* getShapeStyle(ShapeStyle style)
{
    switch (style)
    {
#define STR(r) case ShapeStyle::r: return (char*)#r
        STR(Stroked);
        STR(Filled);
        STR(Outlined);
#undef STR
        default: return (char*)"unknown";
    }
}

constexpr char* getMarkerShape(MarkerOptions::Shape shape)
{
    switch (shape)
    {
#define STR(r) case MarkerOptions::Shape::r: return (char*)#r
        STR(Disc);
        STR(Square);
        STR(Triangle);
        STR(Diamond);
        STR(Chevron);
        STR(Clover);
        STR(Ring);
        STR(Tag);
        STR(Cross);
        STR(Asterisk);
        STR(Infinity);
        STR(Pin);
        STR(Ellipse);
        STR(ArrowBlock);
        STR(ArrowCurved);
        STR(ArrowStealth);
        STR(ArrowTriangle);
        STR(ArrowAngle);
#undef STR
        default: return (char*)"unknown";
    }
}

VkExtent3D getVulkanExtent(Layout extent);

VkFormat getVulkanFormat(FormatDescription desc);

VkImageType getImageType(Layout extent);

VkDeviceSize getSourceSize(AllocHandle alloc);
VkDeviceMemory getMemoryVulkan(AllocHandle alloc);
cudaExternalMemory_t getMemoryCuda(AllocHandle alloc);
// Return the CUDA-mapped device pointer for a LinearAlloc (the interop buffer's CUDA alias), or
// nullptr for the monostate/OpaqueAlloc cases (which carry no persisted mapped pointer).
void *getDevicePtrCuda(AllocHandle alloc);

// Return Vulkan image tiling (linear/opaque) from the allocation type.
VkImageTiling getImageTiling(AllocHandle alloc);

// Query whether the attribute has a defined indexing scheme
// Returns true if indexing contains a source, and false otherwise.
bool hasIndexing(const AttributeDescription& desc);

} // namespace mimir