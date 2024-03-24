#include <optix_device.h>

#include "DepthLaunchParams.h"
#include "Device/Camera.h"
#include "UniUtils/ConversionUtils.h"

using namespace EasyRender;

extern "C" __constant__ DepthLaunchParams param;

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

    float result;
    std::uint32_t u0, u1;
    PackPointer(result, u0, u1);

    glm::vec3 rayDir = PinholeGenerateRay(
        { idx_x, idx_y }, param.fbSize, param.camera);

    optixTrace(param.traversable, UniUtils::ToFloat3(param.camera.pos),
               UniUtils::ToFloat3(rayDir), 1e-5, 1e30, 0, 255,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, RADIANCE_TYPE, RAY_TYPE_COUNT,
               RADIANCE_TYPE, u0, u1);

    auto idx = (std::size_t)optixGetLaunchDimensions().x * idx_y + idx_x;

    // if (result>0) printf("%f\n", result);
    param.depthBuffer[idx] = result;
    if (param.frameID > 1)
    {
        float t = (result - param.minDepth) / (param.maxDepth - param.minDepth);
        unsigned int color;
        if (t < 0)
            color = 0;
        else
            color = 50 + (1 - t) * 200;
        param.colorBuffer[idx].r = color;
        param.colorBuffer[idx].g = color;
        param.colorBuffer[idx].b = color;
        param.colorBuffer[idx].a = 0xFF;
    }
}

extern "C" __global__ void __miss__radiance()
{
    auto &result =
        UnpackPointer<float>(optixGetPayload_0(), optixGetPayload_1());
    result = -1;
}

extern "C" __global__ void __closesthit__radiance()
{
    auto &result =
        UnpackPointer<float>(optixGetPayload_0(), optixGetPayload_1());
    result = optixGetRayTmax();
}

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */
}