#pragma once

#include <cuda_runtime_api.h>
#include <array> // std::array

namespace mimir
{

// Specifies which cuda resource is mapped to the view
enum class ResourceType  { Buffer, IndexBuffer, Texture, LinearTexture };
// Specifies the number of spatial dimensions of the view
enum class DataDomain    { Domain2D, Domain3D };
// Specifies the data layout
enum class DomainType    { Structured, Unstructured };
// Specifies the type of view that will be visualized
enum class ViewType      { Markers, Edges, Voxels, Image };
// Specifies the DataType stored in the texture corresponding to a view
// TODO: Change name to ComponentType
enum class DataType      { Int, Long, Short, Char, Float, Double };
enum class DataLayout    { Layout1D, Layout2D, Layout3D };
enum class AttributeType { Position, Color, Size, Index };

union DataSize
{
    int x;
    int2 xy;
    int3 xyz;
};

static std::array<ResourceType, 4> kAllResources = {
    ResourceType::Buffer,
    ResourceType::IndexBuffer,
    ResourceType::Texture,
    ResourceType::LinearTexture
};
static std::array<DomainType, 2> kAllDomains = {
    DomainType::Structured,
    DomainType::Unstructured
};
static std::array<ViewType, 4> kAllViewTypes = {
    ViewType::Markers,
    ViewType::Edges,
    ViewType::Voxels,
    ViewType::Image
};
static std::array<DataType, 6> kAllDataTypes = {
    DataType::Int,
    DataType::Long,
    DataType::Short,
    DataType::Char,
    DataType::Float,
    DataType::Double
};

constexpr size_t getDataSize(DataType t, unsigned channel_count)
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
#define STR(r) case DataType::r: return (char*)#r
        STR(Int);
        STR(Long);
        STR(Short);
        STR(Char);
        STR(Float);
        STR(Double);
#undef STR
        default: return (char*)"unknown";
    }
}

constexpr char* getDataDomain(DataDomain d)
{
    switch (d)
    {
        case DataDomain::Domain2D: return (char*)"2D";
        case DataDomain::Domain3D: return (char*)"3D";
        default: return (char*)"unknown";
    }
}

constexpr char* getDomainType(DomainType t)
{
    switch (t)
    {
#define STR(r) case DomainType::r: return (char*)#r
        STR(Structured);
        STR(Unstructured);
#undef STR
        default: return (char*)"unknown";
    }
}

constexpr char* getResourceType(ResourceType t)
{
    switch (t)
    {
#define STR(r) case ResourceType::r: return (char*)#r
        STR(Buffer);
        STR(IndexBuffer);
        STR(Texture);
        STR(LinearTexture);
#undef STR
        default: return (char*)"unknown";
    }
}

constexpr uint3 getSize(DataSize size, DataLayout layout)
{
    uint3 sz = {1, 1, 1};
    switch (layout)
    {
        case DataLayout::Layout1D:
        {
            sz.x = size.x;
            break;
        }
        case DataLayout::Layout2D:
        {
            sz.x = size.xy.x;
            sz.y = size.xy.y;
            break;
        }
        case DataLayout::Layout3D:
        {
            sz.x = size.xyz.x;
            sz.y = size.xyz.y;
            sz.z = size.xyz.z;
            break;
        }
        default: break;
    }
    return sz;
};

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

constexpr char* getDataLayout(DataLayout l)
{
    switch (l)
    {
        case DataLayout::Layout1D: return (char*)"1D";
        case DataLayout::Layout2D: return (char*)"2D";
        case DataLayout::Layout3D: return (char*)"3D";
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
        STR(Index);
#undef STR
        default: return (char*)"unknown";
    }
}

constexpr char* getViewType(ViewType type)
{
    switch (type)
    {
#define STR(r) case ViewType::r: return (char*)#r
        STR(Markers);
        STR(Edges);
        STR(Voxels);
        STR(Image);
#undef STR
        default: return (char*)"unknown";
    }
}

} // namespace mimir