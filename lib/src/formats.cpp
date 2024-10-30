#include "internal/api.hpp"

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include <type_traits>

namespace mimir
{

uint32_t getFormatSize(std::span<const FormatDescription> formats)
{
    uint32_t sz = 0;
    for (const auto& format : formats) { sz += format.getSize(); }
    return sz;
}

VkImageTiling getImageTiling(AllocationType type)
{
    switch (type)
    {
        case AllocationType::Linear: { return VK_IMAGE_TILING_LINEAR; }
        case AllocationType::Opaque:
        default:                     { return VK_IMAGE_TILING_OPTIMAL; }
    }
}

VkImageType getImageType(ViewExtent extent)
{
    int dim_count = (extent.x > 1) + (extent.y > 1) + (extent.z > 1);
    switch (dim_count)
    {
        case 1: { return VK_IMAGE_TYPE_1D; }
        case 2: { return VK_IMAGE_TYPE_2D; }
        case 3: default: { return VK_IMAGE_TYPE_3D; }
    }
}

// Converts a memory format description to its Vulkan equivalent.
VkFormat getVulkanFormat(FormatDescription desc)
{
    #define FORMAT(n,suffix) switch (desc.components) \
    { \
        case 1: return VK_FORMAT_R ## n ## _ ## suffix; \
        case 2: return VK_FORMAT_R ## n ## G ## n ## _ ## suffix; \
        case 3: return VK_FORMAT_R ## n ## G ## n ## B ## n ## _ ## suffix; \
        case 4: return VK_FORMAT_R ## n ## G ## n ## B ## n ## A ## n ## _ ## suffix; \
        default: return VK_FORMAT_UNDEFINED; \
    }
    switch (desc.kind)
    {
        case FormatKind::Float: switch (desc.size)
        {
            case 4: FORMAT(32, SFLOAT) // 32 bits
            case 8: FORMAT(64, SFLOAT) // 64 bits
            case 2: FORMAT(16, SFLOAT) // 16 bits
            default: return VK_FORMAT_UNDEFINED;
        }
        case FormatKind::Signed: switch (desc.size)
        {
            case 4: FORMAT(32, SINT)
            case 8: FORMAT(64, SINT)
            case 2: FORMAT(16, SINT)
            case 1: FORMAT(8, SINT)
            default: return VK_FORMAT_UNDEFINED;
        }
        case FormatKind::Unsigned: switch (desc.size)
        {
            case 4: FORMAT(32, UINT)
            case 8: FORMAT(64, UINT)
            case 2: FORMAT(16, UINT)
            case 1: FORMAT(8, UINT)
            default: return VK_FORMAT_UNDEFINED;
        }
        case FormatKind::SignedNormalized: switch (desc.size)
        {
            case 2: FORMAT(16, SNORM)
            case 1: FORMAT(8, SNORM)
            default: return VK_FORMAT_UNDEFINED;
        }
        case FormatKind::UnsignedNormalized: switch (desc.size)
        {
            case 2: FORMAT(16, UNORM)
            case 1: FORMAT(8, UNORM)
            default: return VK_FORMAT_UNDEFINED;
        }
        default: return VK_FORMAT_UNDEFINED;
    }
    #undef FORMAT
}

template <typename T> FormatKind getFormatKind()
{
    if constexpr (std::is_floating_point_v<T>) return FormatKind::Float;
    if constexpr (std::is_signed_v<T>) return FormatKind::Signed;
    if constexpr (std::is_unsigned_v<T>) return FormatKind::Unsigned;
}

template <typename T, int N> FormatDescription buildFormat()
{
    FormatKind kind = getFormatKind<T>();
    return { .kind = kind, .size = sizeof(T), .components = N };
}

#ifndef uint
#define uint unsigned int
#endif

#ifndef uchar
#define uchar unsigned char
#endif

#ifndef ushort
#define ushort unsigned short
#endif

#ifndef ulong
#define ulong unsigned long
#endif

#define SPECIALIZE(T) template <> std::vector<FormatDescription> \
FormatDescription::make<T>() { return { buildFormat<T,1>() }; }
    SPECIALIZE(float);
    SPECIALIZE(double);
    SPECIALIZE(char);
    SPECIALIZE(short);
    SPECIALIZE(int);
    SPECIALIZE(long);
    SPECIALIZE(uchar);
    SPECIALIZE(ushort);
    SPECIALIZE(uint);
    SPECIALIZE(ulong);
#undef SPECIALIZE

#define SPECIALIZE_VEC(T,N) template <> std::vector<FormatDescription> \
FormatDescription::make<T##N>() { return { buildFormat<T,N>() }; }
    SPECIALIZE_VEC(float, 2);
    SPECIALIZE_VEC(float, 3);
    SPECIALIZE_VEC(float, 4);
    SPECIALIZE_VEC(double, 2);
    SPECIALIZE_VEC(double, 3);
    SPECIALIZE_VEC(double, 4);
    SPECIALIZE_VEC(char, 2);
    SPECIALIZE_VEC(char, 3);
    SPECIALIZE_VEC(char, 4);
    SPECIALIZE_VEC(short, 2);
    SPECIALIZE_VEC(short, 3);
    SPECIALIZE_VEC(short, 4);
    SPECIALIZE_VEC(int, 2);
    SPECIALIZE_VEC(int, 3);
    SPECIALIZE_VEC(int, 4);
    SPECIALIZE_VEC(long, 2);
    SPECIALIZE_VEC(long, 3);
    SPECIALIZE_VEC(long, 4);
    SPECIALIZE_VEC(uchar, 2);
    SPECIALIZE_VEC(uchar, 3);
    SPECIALIZE_VEC(uchar, 4);
    SPECIALIZE_VEC(ushort, 2);
    SPECIALIZE_VEC(ushort, 3);
    SPECIALIZE_VEC(ushort, 4);
    SPECIALIZE_VEC(uint, 2);
    SPECIALIZE_VEC(uint, 3);
    SPECIALIZE_VEC(uint, 4);
    SPECIALIZE_VEC(ulong, 2);
    SPECIALIZE_VEC(ulong, 3);
    SPECIALIZE_VEC(ulong, 4);
#undef SPECIALIZE_VEC

#ifdef uint
#undef uint
#endif

#ifdef uchar
#undef uchar
#endif

#ifdef ushort
#undef ushort
#endif

#ifdef ulong
#undef ulong
#endif


} // namespace mimir