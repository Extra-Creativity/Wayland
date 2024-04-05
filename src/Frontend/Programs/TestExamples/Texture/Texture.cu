#include <optix_device.h>

#include "Device/Camera.h"
#include "Device/Common.h"

#include "TextureLaunchParams.h"
#include "UniUtils/ConversionUtils.h"

using namespace EasyRender;
using namespace EasyRender::Device;
using namespace EasyRender::Programs::Texture;

extern "C" __constant__ Programs::Texture::LaunchParams param;

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

    glm::vec3 result{ 0.0, 0.0, 0.0 };
    std::uint32_t u0, u1;
    PackPointer(result, u0, u1);

    // Normally we need a scale to shift the ray direction, here just omit it.
    glm::vec3 rayDir =
        PinholeGenerateRay({ idx_x, idx_y }, param.fbSize, param.camera);

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
        UnpackPointer<glm::vec3>(optixGetPayload_0(), optixGetPayload_1());
    result = reinterpret_cast<MissData *>(optixGetSbtDataPointer())->bg_color;
}

extern "C" __global__ void __closesthit__radiance()
{
    auto &result =
        UnpackPointer<glm::vec3>(optixGetPayload_0(), optixGetPayload_1());

    const int primID = optixGetPrimitiveIndex();
    glm::vec2 barcentrics = { optixGetTriangleBarycentrics().x,
                              optixGetTriangleBarycentrics().y };

    HitData *mat = reinterpret_cast<HitData *>(optixGetSbtDataPointer());
    glm::ivec3 indices = mat->index[primID];

    if (mat->hasTexture)
    {
        glm::vec2 UV[4] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };
        glm::vec2 tc =
            BarycentricByIndices(UV, indices, barcentrics);
         glm::vec4 texColor = UniUtils::ToVec4<glm::vec4>(
             tex2D<float4>(mat->texture, tc.x, tc.y));
         result = texColor;
    }
    else
    {
        result = mat->Kd;
    }
}

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */
}