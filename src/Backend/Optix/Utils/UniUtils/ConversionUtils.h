#pragma once
#include "cuda_runtime.h"
#include "glm/glm.hpp"

namespace EasyRender::UniUtils
{

template<typename T>
__host__ __device__ __forceinline__ float3 ToFloat3(T vec)
{
    return make_float3(vec.x, vec.y, vec.z);
}

template<typename T>
__host__ __device__ __forceinline__ float4 ToFloat4(T vec)
{
    return make_float3(vec.x, vec.y, vec.z, vec.w);
}

template<typename VecType>
__host__ __device__ __forceinline__ VecType ToVec3(float3 vec)
{
    return { vec.x, vec.y, vec.z };
}

template<typename VecType>
__host__ __device__ __forceinline__ VecType ToVec4(float4 vec)
{
    return { vec.x, vec.y, vec.z, vec.w };
}

} // namespace EasyRender::UniUtils