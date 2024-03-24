#include <optix_device.h>
#include <glm/glm.hpp>

#include "WireFrameLaunchParams.h"
#include "UniUtils/ConversionUtils.h"
#include "Device/Camera.h"

using namespace EasyRender;

extern "C" __constant__ Programs::WireFrame::LaunchParams param;

enum
{
    RADIANCE_TYPE = 0,
    RAY_TYPE_COUNT
};

template<typename T>
__device__ void PackPointer(T &data, std::uint32_t &u0, std::uint32_t &u1)
{
    auto ptr = reinterpret_cast<std::uintptr_t>(&data);
    u1 = ptr, u0 = ptr >> 32;
}

template<typename T>
__device__ T &UnpackPointer(std::uint32_t u0, std::uint32_t u1)
{
    return *reinterpret_cast<T *>(std::uintptr_t{ u0 } << 32 | u1);
}


extern "C" __global__ void __raygen__RenderFrame()
{
    auto idx_x = optixGetLaunchIndex().x, idx_y = optixGetLaunchIndex().y;
    auto idx = (std::size_t)optixGetLaunchDimensions().x * idx_y + idx_x;

    float3 result{ 0, 0, 0 };
    std::uint32_t u0, u1;
    PackPointer(result, u0, u1);

    // Normally we need a scale to shift the ray direction, here just omit it.    
    unsigned int seed = tea<4>(idx, param.frameID);
    glm::vec3 rayDir =
        PinholeGenerateRay({ idx_x, idx_y }, param.fbSize, param.camera, seed);

    optixTrace(param.traversable, UniUtils::ToFloat3(param.camera.pos),
               UniUtils::ToFloat3(rayDir), 1e-5, 1e30, 0, 255,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, RADIANCE_TYPE, RAY_TYPE_COUNT,
               RADIANCE_TYPE, u0, u1);

    param.colorBuffer[idx].r = result.x * 255;
    param.colorBuffer[idx].g = result.y * 255;
    param.colorBuffer[idx].b = result.z * 255;
    param.colorBuffer[idx].a = 0xFF;
}

extern "C" __global__ void __miss__radiance()
{
    auto &result =
        UnpackPointer<float3>(optixGetPayload_0(), optixGetPayload_1());
    result = { 0.0, 0.0, 0.0 };
}

extern "C" __global__ void __closesthit__radiance()
{
    auto &result =
        UnpackPointer<float3>(optixGetPayload_0(), optixGetPayload_1());

    float2 w = optixGetTriangleBarycentrics();

    float e = 0.01;
    if (w.x < e || w.x > 1 - e || w.y < e || w.y > 1-e)
        result = { 1, 0, 0 };
    else
    {
        result = { 0.9, 0.9, 0.9 };
    }
}

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */
}