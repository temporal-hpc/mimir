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

VkFormat getVulkanFormat(FormatDescription desc)
{
    switch (desc.kind)
    {
        case FormatKind::Float: switch (desc.size)
        {
            case 4: switch (desc.components) // 32 bits
            {
                case 1: return VK_FORMAT_R32_SFLOAT;
                case 2: return VK_FORMAT_R32G32_SFLOAT;
                case 3: return VK_FORMAT_R32G32B32_SFLOAT;
                case 4: return VK_FORMAT_R32G32B32A32_SFLOAT;
                default: return VK_FORMAT_UNDEFINED;
            }
            case 8: switch (desc.components) // 64 bits
            {
                case 1: return VK_FORMAT_R64_SFLOAT;
                case 2: return VK_FORMAT_R64G64_SFLOAT;
                case 3: return VK_FORMAT_R64G64B64_SFLOAT;
                case 4: return VK_FORMAT_R64G64B64A64_SFLOAT;
                default: return VK_FORMAT_UNDEFINED;
            }
            case 2: switch (desc.components) // 16 bits
            {
                case 1: return VK_FORMAT_R16_SFLOAT;
                case 2: return VK_FORMAT_R16G16_SFLOAT;
                case 3: return VK_FORMAT_R16G16B16_SFLOAT;
                case 4: return VK_FORMAT_R16G16B16A16_SFLOAT;
                default: return VK_FORMAT_UNDEFINED;
            }
            default: return VK_FORMAT_UNDEFINED;
        }
        case FormatKind::Signed: switch (desc.size)
        {
            case 4: switch (desc.components) // 32 bits
            {
                case 1: return VK_FORMAT_R32_SINT;
                case 2: return VK_FORMAT_R32G32_SINT;
                case 3: return VK_FORMAT_R32G32B32_SINT;
                case 4: return VK_FORMAT_R32G32B32A32_SINT;
                default: return VK_FORMAT_UNDEFINED;
            }
            case 8: switch (desc.components) // 64 bits
            {
                case 1: return VK_FORMAT_R64_SINT;
                case 2: return VK_FORMAT_R64G64_SINT;
                case 3: return VK_FORMAT_R64G64B64_SINT;
                case 4: return VK_FORMAT_R64G64B64A64_SINT;
                default: return VK_FORMAT_UNDEFINED;
            }
            case 2: switch (desc.components) // 16 bits
            {
                case 1: return VK_FORMAT_R16_SINT;
                case 2: return VK_FORMAT_R16G16_SINT;
                case 3: return VK_FORMAT_R16G16B16_SINT;
                case 4: return VK_FORMAT_R16G16B16A16_SINT;
                default: return VK_FORMAT_UNDEFINED;
            }
            case 1: switch (desc.components) // 8 bits
            {
                case 1: return VK_FORMAT_R8_SINT;
                case 2: return VK_FORMAT_R8G8_SINT;
                case 3: return VK_FORMAT_R8G8B8_SINT;
                case 4: return VK_FORMAT_R8G8B8A8_SINT;
                default: return VK_FORMAT_UNDEFINED;
            }
            default: return VK_FORMAT_UNDEFINED;
        }
        case FormatKind::Unsigned: switch (desc.size)
        {
            case 4: switch (desc.components) // 32 bits
            {
                case 1: return VK_FORMAT_R32_UINT;
                case 2: return VK_FORMAT_R32G32_UINT;
                case 3: return VK_FORMAT_R32G32B32_UINT;
                case 4: return VK_FORMAT_R32G32B32A32_UINT;
                default: return VK_FORMAT_UNDEFINED;
            }
            case 8: switch (desc.components) // 64 bits
            {
                case 1: return VK_FORMAT_R64_UINT;
                case 2: return VK_FORMAT_R64G64_UINT;
                case 3: return VK_FORMAT_R64G64B64_UINT;
                case 4: return VK_FORMAT_R64G64B64A64_UINT;
                default: return VK_FORMAT_UNDEFINED;
            }
            case 2: switch (desc.components) // 16 bits
            {
                case 1: return VK_FORMAT_R16_UINT;
                case 2: return VK_FORMAT_R16G16_UINT;
                case 3: return VK_FORMAT_R16G16B16_UINT;
                case 4: return VK_FORMAT_R16G16B16A16_UINT;
                default: return VK_FORMAT_UNDEFINED;
            }
            case 1: switch (desc.components) // 8 bits
            {
                case 1: return VK_FORMAT_R8_UINT;
                case 2: return VK_FORMAT_R8G8_UINT;
                case 3: return VK_FORMAT_R8G8B8_UINT;
                case 4: return VK_FORMAT_R8G8B8A8_UINT;
                default: return VK_FORMAT_UNDEFINED;
            }
            default: return VK_FORMAT_UNDEFINED;
        }
        case FormatKind::SignedNormalized: switch (desc.size)
        {
            case 2: switch (desc.components) // 16 bits
            {
                case 1: return VK_FORMAT_R16_SNORM;
                case 2: return VK_FORMAT_R16G16_SNORM;
                case 3: return VK_FORMAT_R16G16B16_SNORM;
                case 4: return VK_FORMAT_R16G16B16A16_SNORM;
                default: return VK_FORMAT_UNDEFINED;
            }
            case 1: switch (desc.components) // 8 bits
            {
                case 1: return VK_FORMAT_R8_SNORM;
                case 2: return VK_FORMAT_R8G8_SNORM;
                case 3: return VK_FORMAT_R8G8B8_SNORM;
                case 4: return VK_FORMAT_R8G8B8A8_SNORM;
                default: return VK_FORMAT_UNDEFINED;
            }
            default: return VK_FORMAT_UNDEFINED;
        }
        case FormatKind::UnsignedNormalized: switch (desc.size)
        {
            case 2: switch (desc.components) // 16 bits
            {
                case 1: return VK_FORMAT_R16_UNORM;
                case 2: return VK_FORMAT_R16G16_UNORM;
                case 3: return VK_FORMAT_R16G16B16_UNORM;
                case 4: return VK_FORMAT_R16G16B16A16_UNORM;
                default: return VK_FORMAT_UNDEFINED;
            }
            case 1: switch (desc.components) // 8 bits
            {
                case 1: return VK_FORMAT_R8_UNORM;
                case 2: return VK_FORMAT_R8G8_UNORM;
                case 3: return VK_FORMAT_R8G8B8_UNORM;
                case 4: return VK_FORMAT_R8G8B8A8_UNORM;
                default: return VK_FORMAT_UNDEFINED;
            }
            default: return VK_FORMAT_UNDEFINED;
        }
        default: return VK_FORMAT_UNDEFINED;
    }
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

#define SPECIALIZE(T) template <> std::vector<FormatDescription> \
FormatDescription::make<T>() { return { buildFormat<T,1>() }; }
    SPECIALIZE(int);
    SPECIALIZE(float);
    SPECIALIZE(double);
    SPECIALIZE(uint);
    SPECIALIZE(uchar);
#undef SPECIALIZE

#define SPECIALIZE_VEC(T,N) template <> std::vector<FormatDescription> \
FormatDescription::make<T##N>() { return { buildFormat<T,N>() }; }
    SPECIALIZE_VEC(int, 2);
    SPECIALIZE_VEC(int, 3);
    SPECIALIZE_VEC(int, 4);
    SPECIALIZE_VEC(float, 2);
    SPECIALIZE_VEC(float, 3);
    SPECIALIZE_VEC(float, 4);
    SPECIALIZE_VEC(double, 2);
    SPECIALIZE_VEC(double, 3);
    SPECIALIZE_VEC(double, 4);
    SPECIALIZE_VEC(uint, 2);
    SPECIALIZE_VEC(uint, 3);
    SPECIALIZE_VEC(uint, 4);
    SPECIALIZE_VEC(uchar, 2);
    SPECIALIZE_VEC(uchar, 3);
    SPECIALIZE_VEC(uchar, 4);
#undef SPECIALIZE_VEC

#ifdef uint
#undef uint
#endif

#ifdef uchar
#undef uchar
#endif

} // namespace mimir