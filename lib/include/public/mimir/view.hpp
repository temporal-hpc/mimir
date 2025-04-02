#pragma once

#include <cuda_runtime_api.h>

#include <map> // std::map
#include <variant> // std::variant
#include <vector> // std::vector

namespace mimir
{

// Forward declarations
struct LinearAlloc;
struct OpaqueAlloc;
typedef std::variant<std::monostate, LinearAlloc*, OpaqueAlloc*> AllocHandle;

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
    // Format description helper for commonly used types, including CUDA vector types.
    template <typename T> static FormatDescription make();
};

enum class AttributeType { Position, Color, Size, Rotation, Texcoord };

struct IndexDescription
{
    // Handle for the allocation containing the index data.
    // Must be non-null to describe a valid indexing.
    AllocHandle source = {};
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
    AllocHandle source = {};
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
    AllocHandle source = {};
    // Format description for texels.
    FormatDescription format = {};
    // Amount of texels in (width, height, depth) format.
    Layout extent = {};
    // Number of mipmap levels stored in the texture.
    unsigned int levels = 1;
};

enum class ShapeStyle { Stroked, Filled, Outlined };

struct MarkerOptions
{
    enum class Shape {
        // Disc-based
        Disc, Clover, Ring, Infinity, Pin,
        // Square-based
        Square, Diamond, Chevron,
        // Half-plane based
        Triangle, Tag, Cross, Asterisk, ArrowBlock,
        // Styled arrows (angled arrows fixed at 60 deg for now)
        ArrowCurved, ArrowStealth, ArrowTriangle, ArrowAngle,
        // Special
        Ellipse,
    };

    // Marker shape used in this view.
    Shape shape;

    // Initialize options with sensible defaults when not specified by the user.
    static MarkerOptions defaults() {
        return { .shape = Shape::Disc };
    }
};

struct MeshOptions {
    bool periodic;

    static MeshOptions defaults() {
        return { .periodic = false };
    }
};

// Structure for specifing view-specific options matching the selected view type.
// If left uninitialized or initialized for the wrong view type,
// a default configuration will be initialized and used.
typedef std::variant<std::monostate, MarkerOptions, MeshOptions> ViewOptions;

struct ViewDescription
{
    // Determines the element to display in the current view.
    ViewType type;
    // Additional configuration parameters for the view, depending on its type.
    // This pointer must reference an extension structure matching
    // the view type specified above, or else it will be ignored.
    ViewOptions options;
    // Determines whether to draw 2D or 3D elements for the given view type.
    DomainType domain;
    // Spatial extent of the positions represented in the view.
    // Dictionary of attached attributes.
    std::map<AttributeType, AttributeDescription> attributes;
    // Spatial layout of the view.
    Layout layout;
    // Shape style (stroked, filled, outlined, etc.) used to draw elements in this view.
    ShapeStyle style     = ShapeStyle::Filled;
    // Whether the view contents are shown or hidden during display.
    // Created views are visible by default, which can be changed at runtime
    // through the toggleViewVisibility() function.
    bool visible         = true;
    // Default color for view elements in RGBA format.
    // This value is used when no color attribute is specified.
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