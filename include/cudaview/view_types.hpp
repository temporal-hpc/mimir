#pragma once

enum class PresentOptions { Immediate, TripleBuffering, VSync };

// Specifies the number of spatial dimensions of the view
enum class DataDomain    { Domain2D, Domain3D };
// Specifies the data layout
enum class ResourceType  { UnstructuredBuffer, StructuredBuffer, Texture, TextureLinear };
// Specifies the type of primitive that will be visualized 
enum class PrimitiveType { Points, Edges, Voxels };
// Specifies the DataType stored in the texture corresponding to a view
enum class DataType { Int, Float, Char };

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
        case DataType::Int:   return "Int";
        case DataType::Float: return "Float";
        case DataType::Char:  return "Char";     
        default: return "unknown";
    }
}