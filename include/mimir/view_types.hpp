#pragma once

#include <array> // std::array

namespace mimir
{

// Specifies which cuda resource is mapped to the view
enum class ResourceType { Buffer, IndexBuffer, Texture };
// Specifies the number of spatial dimensions of the view
enum class DataDomain   { Domain2D, Domain3D };
// Specifies the data layout
enum class DomainType   { Structured, Unstructured };
// Specifies the type of primitive that will be visualized 
enum class ElementType  { Markers, Edges, Voxels, Image };
// Specifies the DataType stored in the texture corresponding to a view
// TODO: Change name to ComponentType
enum class DataType     { Int, Long, Short, Char, Float, Double };

constexpr size_t getDataSize(DataType t, uint channel_count)
{
    switch (t)
    {
        case DataType::Int:    return sizeof(int) * channel_count;
        case DataType::Long:   return sizeof(long) * channel_count;
        case DataType::Short:  return sizeof(short) * channel_count;
        case DataType::Char:   return sizeof(char) * channel_count;
        case DataType::Float:  return sizeof(float) * channel_count;
        case DataType::Double: return sizeof(double) * channel_count;
        default: return 0;
    }
}

constexpr char* getDataType(DataType type)
{
    switch (type)
    {
#define STR(r) case DataType::r: return #r
        STR(Int);
        STR(Long);
        STR(Short);
        STR(Char);
        STR(Float);
        STR(Double);
#undef STR
        default: return "unknown";
    }
}

constexpr char* getDataDomain(DataDomain d)
{
    switch (d)
    {
        case DataDomain::Domain2D: return "2D";
        case DataDomain::Domain3D: return "3D";
        default: return "unknown";
    }
}

constexpr char* getDomainType(DomainType t)
{
    switch (t)
    {
#define STR(r) case DomainType::r: return #r
        STR(Structured);
        STR(Unstructured);
#undef STR
        default: return "unknown";
    }
}

constexpr char* getResourceType(ResourceType t)
{
    switch (t)
    {
#define STR(r) case ResourceType::r: return #r
        STR(Buffer);
        STR(IndexBuffer);
        STR(Texture);
#undef STR
        default: return "unknown";
    }
}

constexpr char* getElementType(ElementType t)
{
    switch (t)
    {
#define STR(r) case ElementType::r: return #r
        STR(Markers);
        STR(Edges);
        STR(Voxels);
        STR(Image);
#undef STR
        default: return "unknown";
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

enum class AttributeType { Position, Color, Size, Index };
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
        STR(Index);
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

} // namespace mimir