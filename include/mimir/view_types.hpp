#pragma once

namespace mimir
{

// Specifies which cuda resource is mapped to the view
enum class ResourceType { Buffer, Texture };
// Specifies the number of spatial dimensions of the view
enum class DataDomain   { Domain2D, Domain3D };
// Specifies the data layout
enum class DomainType   { Structured, Unstructured };
// Specifies the type of primitive that will be visualized 
enum class ElementType  { Markers, Edges, Voxels, Texels };
// Specifies the DataType stored in the texture corresponding to a view
enum class DataType     { Int, Float, Char };

constexpr size_t getDataSize(DataType t, uint channel_count)
{
    switch (t)
    {
        case DataType::Int:   return sizeof(int) * channel_count;
        case DataType::Float: return sizeof(float) * channel_count;
        case DataType::Char:  return sizeof(char) * channel_count;
        default: return 0;
    }
}

constexpr char* getDataType(DataType type)
{
    switch (type)
    {
#define STR(r) case DataType::r: return #r
        STR(Int);
        STR(Float);
        STR(Char);
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
        STR(Texels);
#undef STR
        default: return "unknown";
    }
}

} // namespace mimir