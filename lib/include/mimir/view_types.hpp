#pragma once

#include <cuda_runtime_api.h>
#include <cuda_fp16.h> // half
#include <vulkan/vulkan.h>

namespace mimir
{

// Specifies the type of view that will be visualized
enum class ViewType      { Markers, Edges, Voxels, Image };
// Specifies the data type stored in the texture corresponding to a view
enum class ComponentType { Int, Long, Short, Char, Float, Double, Half };
// Specifies which cuda resource is mapped
enum class ResourceType  { Buffer, IndexBuffer, Texture, LinearTexture };
// Specifies the number of spatial dimensions in the view
enum class DataDomain    { Domain2D, Domain3D };
// Specifies the data layout
enum class DataLayout    { Layout1D, Layout2D, Layout3D };
enum class DomainType    { Structured, Unstructured };
enum class AttributeType { Position, Color, Size, Index };

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

constexpr char* getComponentType(ComponentType type)
{
    switch (type)
    {
#define STR(r) case ComponentType::r: return (char*)#r
        STR(Int);
        STR(Long);
        STR(Short);
        STR(Char);
        STR(Float);
        STR(Double);
        STR(Half);
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

constexpr unsigned getSize(uint3 size, DataLayout layout)
{
    switch (layout)
    {
        case DataLayout::Layout1D: return size.x;
        case DataLayout::Layout2D: return size.x * size.y;
        case DataLayout::Layout3D: return size.x * size.y * size.z;
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

constexpr size_t getElementCount(uint3 size, DataLayout layout)
{
    switch (layout)
    {
        case DataLayout::Layout1D: return size.x;
        case DataLayout::Layout2D: return size.x * size.y;
        case DataLayout::Layout3D: return size.x * size.y * size.z;
        default: return 0;
    }
};

constexpr size_t getComponentSize(ComponentType t)
{
    switch (t)
    {
        case ComponentType::Int:    return sizeof(int);
        case ComponentType::Long:   return sizeof(long);
        case ComponentType::Short:  return sizeof(short);
        case ComponentType::Char:   return sizeof(char);
        case ComponentType::Float:  return sizeof(float);
        case ComponentType::Double: return sizeof(double);
        case ComponentType::Half:   return sizeof(half);
        default: return 0;
    }
}

constexpr size_t getBytesize(ComponentType t, unsigned channel_count)
{
    return getComponentSize(t) * channel_count;
}

// Converts an interop memory data type to its Vulkan format equivalent
constexpr VkFormat getDataFormat(ComponentType type, unsigned channel_count)
{
    switch (type)
    {
        case ComponentType::Int: switch (channel_count)
        {
            case 1: return VK_FORMAT_R32_SINT;
            case 2: return VK_FORMAT_R32G32_SINT;
            case 3: return VK_FORMAT_R32G32B32_SINT;
            case 4: return VK_FORMAT_R32G32B32A32_SINT;
            default: return VK_FORMAT_UNDEFINED;
        }
        case ComponentType::Char: switch (channel_count)
        {
            case 1: return VK_FORMAT_R8_SRGB;
            case 2: return VK_FORMAT_R8G8_SRGB;
            case 3: return VK_FORMAT_R8G8B8_SRGB;
            case 4: return VK_FORMAT_R8G8B8A8_SRGB;
            default: return VK_FORMAT_UNDEFINED;
        }
        case ComponentType::Float: switch (channel_count)
        {
            case 1: return VK_FORMAT_R32_SFLOAT;
            case 2: return VK_FORMAT_R32G32_SFLOAT;
            case 3: return VK_FORMAT_R32G32B32_SFLOAT;
            case 4: return VK_FORMAT_R32G32B32A32_SFLOAT;
            default: return VK_FORMAT_UNDEFINED;
        }
        case ComponentType::Double: switch (channel_count)
        {
            case 1: return VK_FORMAT_R64_SFLOAT;
            case 2: return VK_FORMAT_R64G64_SFLOAT;
            case 3: return VK_FORMAT_R64G64B64_SFLOAT;
            case 4: return VK_FORMAT_R64G64B64A64_SFLOAT;
            default: return VK_FORMAT_UNDEFINED;
        }
        case ComponentType::Half: switch (channel_count)
        {
            case 1: return VK_FORMAT_R16_SFLOAT;
            case 2: return VK_FORMAT_R16G16_SFLOAT;
            case 3: return VK_FORMAT_R16G16B16_SFLOAT;
            case 4: return VK_FORMAT_R16G16B16A16_SFLOAT;
            default: return VK_FORMAT_UNDEFINED;
        }
        default: return VK_FORMAT_UNDEFINED;
    }
}

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

constexpr size_t getBytesize(DataFormat format)
{
    return getComponentSize(format.type) * format.components;
}

} // namespace mimir