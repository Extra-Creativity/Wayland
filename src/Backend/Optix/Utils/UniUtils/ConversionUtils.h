#pragma once
#include "cuda_runtime.h"

namespace Wayland::UniUtils
{

template<typename T>
__host__ __device__ __forceinline__ float3 ToFloat3(T vec)
{
    return make_float3(vec.x, vec.y, vec.z);
}

} // namespace Wayland::UniUtils