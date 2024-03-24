#include <optix_device.h>

#include "Device/Camera.h"
#include "ColorLaunchParams.h"
#include "UniUtils/ConversionUtils.h"

using namespace EasyRender;

extern "C" __constant__ ColorLaunchParams param;

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

/*! helper function that creates a semi-random color from an ID */
inline __device__ float3 randomColor(int i)
{
    int r = unsigned(i) * 13 * 17 + 0x234235;
    int g = unsigned(i) * 7 * 3 * 5 + 0x773477;
    int b = unsigned(i) * 11 * 19 + 0x223766;
    return { (r & 255) / 255.f, (g & 255) / 255.f, (b & 255) / 255.f };
}

extern "C" __global__ void __raygen__RenderFrame()
{
    auto idx_x = optixGetLaunchIndex().x, idx_y = optixGetLaunchIndex().y;
    //  scale to [-1, 1], row goes vertically.
    float xPos = 2.f * idx_x / optixGetLaunchDimensions().x - 1;
    float yPos = 2.f * idx_y / optixGetLaunchDimensions().y - 1;

    float3 result{ 0.8, 0.8, 0.8 };
    std::uint32_t u0, u1;
    PackPointer(result, u0, u1);

    // Normally we need a scale to shift the ray direction, here just omit it.
    glm::vec3 rayDir =
        glm::normalize(param.camera.lookAt + xPos * param.camera.right +
                       yPos * param.camera.up);

    rayDir = PinholeGenerateRay({ idx_x, idx_y }, param.fbSize, param.camera);

    optixTrace(param.traversable, UniUtils::ToFloat3(param.camera.pos),
               UniUtils::ToFloat3(rayDir), 1e-5, 1e30, 0, 255,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, RADIANCE_TYPE, RAY_TYPE_COUNT,
               RADIANCE_TYPE, u0, u1);

    auto idx = (std::size_t)optixGetLaunchDimensions().x * idx_y + idx_x;
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
    int primID = optixGetPrimitiveIndex();
    result = randomColor(primID);
}

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */
}