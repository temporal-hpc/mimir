#pragma once

// Specifies the number of spatial dimensions of the view
enum class DataDomain   { Domain2D, Domain3D };
// Specifies the data layout
enum class ResourceType { UnstructuredBuffer, StructuredBuffer, Texture };
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

constexpr char* getDataDomain(DataDomain x)
{
    switch (x)
    {
        case DataDomain::Domain2D: return "2D";
        case DataDomain::Domain3D: return "3D";
        default: return "unknown";
    }
}

constexpr char* getResourceType(ResourceType x)
{
    switch (x)
    {
#define STR(r) case ResourceType::r: return #r
        STR(UnstructuredBuffer);
        STR(StructuredBuffer);
        STR(Texture);
#undef STR
        default: return "unknown";
    }
}

constexpr char* getElementType(ElementType x)
{
    switch (x)
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