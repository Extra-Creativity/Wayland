#pragma once
#include <optix_device.h>
#include "glm/glm.hpp"
#include "UniUtils/ConversionUtils.h"

namespace EasyRender::Device
{

static __forceinline__ __device__ void *UnpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void *ptr = reinterpret_cast<void *>(uptr);
    return ptr;
}

static __forceinline__ __device__ void PackPointer(void *ptr, uint32_t &i0,
                                                   uint32_t &i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *GetPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(UnpackPointer(u0, u1));
}

static __forceinline__ __device__ glm::vec3 GetHitPosition()
{
    return UniUtils::ToVec3<glm::vec3>(optixGetWorldRayOrigin()) +
             UniUtils::ToVec3<glm::vec3>(optixGetWorldRayDirection()) *
                 optixGetRayTmax();
}

template<typename T, typename U>
__host__ __device__ __forceinline__ T Barycentric(T v1, T v2, T v3, U coord)
{
    return v1 * (1 - coord.x - coord.y) + v2 * coord.x + v3 * coord.y;
}

template<typename T, typename U, typename V>
__host__ __device__ __forceinline__ T BarycentricByIndices(T *v, U indices,
                                                           V coord)
{
    return Barycentric(v[indices.x], v[indices.y], v[indices.z], coord);
}

__host__ __device__ __forceinline__ glm::vec3 NormalToColor(glm::vec3 N)
{
    glm::vec3 v = { 1, 1, 1 };
    return (N+v)/2.0f;
}

__host__ __device__ __forceinline__ float GetTriangleArea(glm::vec3* V, glm::ivec3 I) {
    glm::vec3 a = V[I.x], b = V[I.y], c = V[I.z];
	return 0.5f * glm::length(glm::cross(b - a, c - a));
}

__host__ __device__ __forceinline__ glm::vec3 GetGeometryNormal(glm::vec3 *V,
                                                          glm::ivec3 I)
{
    glm::vec3 a = V[I.x], b = V[I.y], c = V[I.z];
    return glm::normalize(glm::cross(b - a, c - a));
}


__host__ __device__ __forceinline__ float clamp(float a, float lo, float hi)
{
    return a < lo ? lo : (a > hi ? hi : a);
}

__host__ __device__ __forceinline__ float lerp(float a, float b, float w)
{
    return a * (1 - w) + b * w;
}

__host__ __device__ __forceinline__ glm::vec3 clamp(glm::vec3 v, float lo,
                                                   float hi)
{
    return glm::vec3{ clamp(v.x, lo, hi), clamp(v.y, lo, hi),
                      clamp(v.z, lo, hi) };
}

__host__ __device__ __forceinline__ glm::vec3 lerp(glm::vec3 a, glm::vec3 b,
                                                  float w)
{
    return glm::vec3{ lerp(a.x, b.x, w), lerp(a.y, b.y, w), lerp(a.z, b.z, w) };
}

} // namespace EasyRender