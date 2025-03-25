#pragma once

#include <cuda_runtime_api.h>

#include <map> // std::map
#include <vector> // std::vector

namespace mimir
{

// Forward declarations
struct Allocation;
struct Texture;

// Specifies the type of view that will be visualized
enum class ViewType   { Markers, Edges, Image, Boxes, Voxels };
// Specifies the number of spatial dimensions in the view
enum class DomainType { Domain2D, Domain3D };
// Specifies the numeric type of the data described by this format.
enum class FormatKind { Float, Signed, Unsigned, SignedNormalized, UnsignedNormalized, SRGB };

// Description for the spatial layout of an arrangement of visualizable elements,
// whether texture elements (texels) or other graphics primitives.
struct Layout
{
    // Layout size in cartesian (x,y,z) format.
    unsigned int x = 0, y = 1, z = 1;

    // Returns the total number of elements contained in this layout.
    unsigned int getTotalCount() { return x * y * z; };

    // Creates a layout with default value of 1 for Y and Z axes.
    static Layout make(unsigned int x, unsigned int y = 1, unsigned int z = 1)
    {
        return Layout{ .x = x, .y = y, .z = z };
    }
};

// Descriptor structure for interpreting elements contained in an engine allocation.
// Similar to the ChannelFormatDesc structure in the CUDA SDK, but used to describe
// components for both linear arrays and textures.
struct FormatDescription
{
    // Kind of numeric data stored in each channel.
    FormatKind kind = FormatKind::Float;
    // Size in bytes of each channel component.
    int size = 0;
    // Number of data channels; between 1 and 4.
    int components = 1;

    // Returns the size in bytes of a single data element described in this format.
    unsigned int getSize() const { return size * components; }
    // Helper for creating format descriptions for commonly used types, including CUDA vector types.
    template <typename T> static FormatDescription make();
};

enum class AttributeType { Position, Color, Size, Rotation, Texcoord };

struct IndexDescription
{
    // Handle for the allocation containing the index data.
    // Must be non-null to describe a valid indexing.
    Allocation *source = nullptr;
    // Number of elements contained in the source allocation
    // Must be non-zero to describe a valid indexing.
    unsigned int size = 0;
    // Size in bits of the index type stored in the indices allocation.
    // Must be non-zero to describe a valid indexing.
    unsigned int index_size = 0;
};

// The interpretation of the sources within the engine depends on attribute type and whether
// the indices array is set to a non-null value or not.
struct AttributeDescription
{
    // Handle for the allocation containing the source data.
    // Must be non-null to describe a valid attribute.
    Allocation *source = nullptr;
    // Number of elements contained in the source allocation.
    // Must be non-zero to describe a valid attribute.
    unsigned int size = 0;
    // Format description for the elements in the source array; ignored when sources is null.
    FormatDescription format = {};
    // Handle for the allocation containing indices referencing the source array.
    // Can be left uninitialized for direct access to the source.
    IndexDescription indexing = {};
};

struct TextureDescription
{
    // Handle for the allocation holding the texture memory.
    Allocation *source = nullptr;
    // Format description for texels.
    FormatDescription format = {};
    Layout extent = {};
    // Number of mipmap levels stored in the texture.
    unsigned int levels = 1;
};

struct ViewDescription
{
    // Number of elements contained in the view.
    // The amount of vertices consumed per element depends on view type.
    Layout layout;
    // Determines the element to display in the current view.
    ViewType view_type;
    // Determines whether to draw 2D or 3D elements for the given view type.
    DomainType domain_type;
    // Spatial extent of the positions represented in the view.
    // Dictionary of attached attributes.
    std::map<AttributeType, AttributeDescription> attributes;
    // Whether the view contents are shown or not during display.
    // Created views are visible by default, which can be changed at runtime
    // through the toggleViewVisibility() function.
    bool visible         = true;
    // Default color for elements in the view if no color data is specified.
    float4 default_color = {0.f, 0.f, 0.f, 1.f};
    // Default size for elements in the view if no size data is specified.
    float default_size   = 1.f;
    // Line width size.
    float linewidth      = 1.f;
    // Antialias magnitude.
    float antialias      = 0.f;
    // Dataset translation applied to the positions of all its elements.
    float3 position      = {0.f, 0.f, 0.f};
    // Dataset rotation, applied after translation.
    float3 rotation      = {0.f, 0.f, 0.f};
    // Dataset scale, applied after rotation and translation.
    float3 scale         = {1.f, 1.f, 1.f};
};

} // namespace mimir