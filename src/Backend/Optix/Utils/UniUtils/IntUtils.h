#pragma once
#include "cuda_runtime.h"
#include <limits>

namespace EasyRender::UniUtils
{
template<typename T>
__host__ __device__ __forceinline__ auto RoundUpNonNegative(T x, T y) noexcept
{
    static_assert(std::is_unsigned_v<T>, "Rounded value must be unsigned.");
    assert((std::numeric_limits<T>::max)() - y > x);
    return (x / y + (x % y != 0)) * y;
}

} // namespace Wayland::UniUtils
