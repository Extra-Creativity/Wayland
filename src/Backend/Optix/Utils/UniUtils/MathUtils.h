#pragma once
#include "cuda_runtime.h"
#include <concepts>

namespace Wayland::UniUtils
{
template<typename T>
    requires std::unsigned_integral<T>
__host__ __device__ __forceinline__ auto RoundUpNonNegative(T x, T y) noexcept
{
    return (x / y + (x % y != 0)) * y;
}

} // namespace Wayland::UniUtils
