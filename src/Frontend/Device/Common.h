#pragma once
#include <optix_device.h>
#include "glm/glm.hpp"
#include "UniUtils/ConversionUtils.h"

namespace EasyRender
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

} // namespace EasyRender