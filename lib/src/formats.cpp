#include "mimir/api.hpp"

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>
#include <spdlog/spdlog.h>

#include <type_traits>

namespace mimir
{

template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

VkDeviceSize getSourceSize(AllocHandle alloc)
{
    return std::visit(overloaded{
        [](auto) { return VkDeviceSize(0); },
        [](LinearAlloc *arg) { return arg->size; },
        [](OpaqueAlloc *arg) { return arg->size; }
    }, alloc);
}

VkDeviceMemory getMemoryVulkan(AllocHandle alloc)
{
    return std::visit(overloaded{
        [](auto) { return VkDeviceMemory(VK_NULL_HANDLE); },
        [](LinearAlloc *arg) { return arg->vk_mem; },
        [](OpaqueAlloc *arg) { return arg->vk_mem; }
    }, alloc);
}

cudaExternalMemory_t getMemoryCuda(AllocHandle alloc)
{
    return std::visit(overloaded{
        [](auto) { return cudaExternalMemory_t(nullptr); },
        [](LinearAlloc *arg) { return arg->cuda_extmem; },
        [](OpaqueAlloc *arg) { return arg->cuda_extmem; }
    }, alloc);
}

VkImageTiling getImageTiling(AllocHandle alloc)
{
    return std::visit(overloaded{
        [](auto) { spdlog::trace("LINEAR"); return VK_IMAGE_TILING_LINEAR; },
        [](OpaqueAlloc*) { spdlog::trace("OPTIMAL"); return VK_IMAGE_TILING_OPTIMAL; }
    }, alloc);
}

bool hasIndexing(const AttributeDescription& desc)
{
    return std::visit(overloaded{
        [](auto) { return false; },
        [](LinearAlloc*) { return true; }
    }, desc.indexing.source);
}

// Derive Vulkan image type from the view extent dimensions,
// assuming that a dimension exists if its size is greater than 1.
VkImageType getImageType(Layout extent)
{
    int dim_count = (extent.x > 1) + (extent.y > 1) + (extent.z > 1);
    switch (dim_count)
    {
        case 1: { return VK_IMAGE_TYPE_1D; }
        case 2: { return VK_IMAGE_TYPE_2D; }
        case 3: default: { return VK_IMAGE_TYPE_3D; }
    }
}

VkExtent3D getVulkanExtent(Layout extent)
{
    return VkExtent3D
    {
        .width  = extent.x > 0? extent.x : 1,
        .height = extent.y > 0? extent.y : 1,
        .depth  = extent.z > 0? extent.z : 1,
    };
}

// Converts a memory format description to its Vulkan equivalent.
// Only formats whose channels have the same size are currently supported.
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
            case 1: FORMAT(8, SRGB) // TODO: Handle
            default: return VK_FORMAT_UNDEFINED;
        }
        case FormatKind::Unsigned: switch (desc.size)
        {
            case 4: FORMAT(32, UINT)
            case 8: FORMAT(64, UINT)
            case 2: FORMAT(16, UINT)
            case 1: FORMAT(8, SRGB) // TODO: Handle
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
        case FormatKind::SRGB: switch (desc.size)
        {
            case 1: FORMAT(8, SRGB)
            default: return VK_FORMAT_UNDEFINED;
        }
        default: return VK_FORMAT_UNDEFINED;
    }
    #undef FORMAT
}

// Helper function to derive format kind using C++ type traits.
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

// Define shortnames for unsigned types, for use in macro substitutions.
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


#define SPECIALIZE(T) template <> FormatDescription \
FormatDescription::make<T>() { return buildFormat<T,1>(); }
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

#define SPECIALIZE_VEC(T,N) template <> FormatDescription \
FormatDescription::make<T##N>() { return buildFormat<T,N>(); }
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

// Cleanup definitions
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