#include <mimir/view.hpp>

#include <cuda_runtime_api.h>

#include <type_traits>

namespace mimir
{

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

#define SPECIALIZE(T) template <> FormatDescription FormatDescription::make<T>() { return buildFormat<T,1>(); }
    SPECIALIZE(int);
    SPECIALIZE(float);
    SPECIALIZE(double);
    SPECIALIZE(uint);
    SPECIALIZE(uchar);
#undef SPECIALIZE

#define SPECIALIZE_VEC(T,N) template <> FormatDescription FormatDescription::make<T##N>() { return buildFormat<T,N>(); }
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