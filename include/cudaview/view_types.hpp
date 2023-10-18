#pragma once

enum class PresentOptions { Immediate, TripleBuffering, VSync };

// Specifies the number of spatial dimensions of the view
enum class DataDomain    { Domain2D, Domain3D };
// Specifies the data layout
enum class ResourceType  { UnstructuredBuffer, StructuredBuffer, Texture, TextureLinear };
// Specifies the type of primitive that will be visualized 
enum class PrimitiveType { Points, Edges, Voxels };
// Specifies the datatype stored in the texture corresponding to a view
enum class DataType { int1, int2, int3, int4, float1, float2, float3, float4, char1, char2, char3, char4 };

constexpr size_t getDataSize(DataType t)
{
    switch (t)
    {
#define CONVERT(r) case DataType::r: return sizeof(r)        
        CONVERT(int1);
        CONVERT(int2);
        CONVERT(int3);
        CONVERT(int4);
        CONVERT(float1);
        CONVERT(float2);
        CONVERT(float3);
        CONVERT(float4);
        CONVERT(char1);
        CONVERT(char2);
        CONVERT(char3);
        CONVERT(char4);
#undef CONVERT
        default: return 0;
    }
}

constexpr char* getDataType(DataType type)
{
    switch (type)
    {
        case DataType::int1: return "Int1";
        case DataType::int2: return "Int2";
        case DataType::int3: return "Int3";
        case DataType::int4: return "Int4";
        case DataType::float1: case DataType::char1: return "Float1";
        case DataType::float2: case DataType::char2: return "Float2";
        case DataType::float3: case DataType::char3: return "Float3";
        case DataType::float4: case DataType::char4: return "Float4";
        default: return "unknown";
    }
}