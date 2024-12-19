#pragma once

#include <map> // std::map
#include <vector> // std::vector

namespace mimir
{

// Forward declaration
struct Allocation;
struct ViewDetails;
struct Texture;

// Specifies the type of view that will be visualized
enum class ViewType      { Markers, Edges, Image, Boxes, Voxels };
// Specifies the number of spatial dimensions in the view
enum class DomainType    { Domain2D, Domain3D };

enum class FormatKind    { Float, Signed, Unsigned, SignedNormalized, UnsignedNormalized, SRGB };

struct ViewExtent
{
    unsigned int x, y, z;

    static ViewExtent make(unsigned int x, unsigned int y, unsigned int z)
    {
        return { .x = x, .y = y, .z = z };
    }
};

// Descriptor structure for interpreting elements contained in an engine allocation.
// Similar to the ChannelFormatDesc structure in the CUDA SDK, but here it is used to describe
// vector types in both textures and linear arrays.
struct FormatDescription
{
    // Kind of numeric data stored in each channel.
    FormatKind kind;
    // Size in bytes of each channel component.
    int size;
    // Number of data channels; between 1 and 4.
    int components;

    // Returns the size in bytes of a single data element described in this format.
    unsigned int getSize() const { return size * components; }
    // Helper for creating format descriptions for commonly used types, including CUDA vector types.
    template <typename T> static FormatDescription make();
};

enum class AttributeType { Position, Color, Size, Rotation, Texcoord };

// The interpretation of the sources within the engine depends on attribute type and whether
// the indices array is set to a non-null value or not.
struct AttributeDescription
{
    // Handle for the allocation containing the source data.
    Allocation *source;
    // Number of elements contained in the source allocation.
    unsigned int size;
    // Format description for the elements in the source array; ignored when sources is null.
    FormatDescription format;
    // Handle for the allocation containing indices referencing the source array.
    Allocation *indices;
    // Size in bits of the index type stored in the indices allocation.
    // This value is ignored when indices is null.
    int index_size;
};

// TODO: More texture definitions (e.g. mipmap levels)
struct TextureDescription
{
    // Handle for the allocation holding the texture memory.
    Allocation *source;
    // Format description for texels.
    FormatDescription format;
    ViewExtent extent;
    // Number of mipmap levels stored in the texture.
    unsigned int levels;
};

struct ViewDescription
{
    // Number of elements contained in the view.
    // The amount of vertices consumed per element depends on view type.
    unsigned int element_count;
    // Determines the element to display in the current view.
    ViewType view_type;
    // Determines whether to draw 2D or 3D elements for the given view type.
    DomainType domain_type;
    // Spatial extent of the positions represented in the view.
    ViewExtent extent;
    // Dictionary of attached attributes.
    std::map<AttributeType, AttributeDescription> attributes;
};

struct View
{
    ViewDetails *detail;
    bool visible;
    float default_color[4];
    float default_size;

    // Switches view state between visible and invisible; does not modify underlying data.
    bool toggleVisibility()
    {
        visible = !visible;
        return visible;
    }
};

} // namespace mimir