#pragma once

#include <cuda_runtime_api.h>
#include <cuda_fp16.h> // half
#include <vulkan/vulkan.h>

namespace mimir
{

// Specifies the type of view that will be visualized
enum class ViewType      { Markers, Edges, Voxels, Image, Boxes };
// Specifies the number of spatial dimensions in the view
enum class DataDomain    { Domain2D, Domain3D };

enum class AttributeType { Position, Color, Size };

enum class DataType { float32, float64, float16, int32, int64, int8, int16 };

struct DataFormat {
    // Data type used to interpret the memory associated to the data buffer or image
    DataType type  = DataType::float32;
    // Number of values of specified type in the format
    int components = 1;
    // Determines if source values are signed or unsigned
    bool is_signed = true;
    // TODO: Add enum for integral values (normalize, scale, none)
};

constexpr char* getViewType(ViewType type)
{
    switch (type)
    {
#define STR(r) case ViewType::r: return (char*)#r
        STR(Markers);
        STR(Edges);
        STR(Voxels);
        STR(Image);
        STR(Boxes);
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

constexpr char* getDataType(DataType type)
{
    switch (type)
    {
        case DataType::float32: return (char*)"Float";
        case DataType::float64: return (char*)"Double";
        case DataType::float16: return (char*)"Half";
        case DataType::int32:   return (char*)"Int";
        case DataType::int64:   return (char*)"Long";
        case DataType::int8:    return (char*)"Char";
        case DataType::int16:   return (char*)"Short";
        default: return (char*)"unknown";
    }
}

constexpr size_t getComponentSize(DataType t)
{
    switch (t)
    {
        case DataType::float32: return sizeof(float);
        case DataType::float64: return sizeof(double);
        case DataType::float16: return sizeof(half);
        case DataType::int32:   return sizeof(int);
        case DataType::int64:   return sizeof(long);
        case DataType::int16:   return sizeof(short);
        case DataType::int8:    return sizeof(char);
        default: return 0;
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
#undef STR
        default: return (char*)"unknown";
    }
}

constexpr size_t getBytesize(DataFormat format)
{
    return getComponentSize(format.type) * format.components;
}

constexpr VkIndexType getIndexType(DataType type)
{
    switch (type)
    {
        case DataType::int16: return VK_INDEX_TYPE_UINT16;
        case DataType::int32: return VK_INDEX_TYPE_UINT32;
        // TODO: Add VK_INDEX_TYPE_UINT8_EXT for char
        // and VK_INDEX_TYPE_NONE_KHR for default
        default: return VK_INDEX_TYPE_UINT16;
    }
}

} // namespace mimir