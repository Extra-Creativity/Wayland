#include <optix_device.h>

#include "Device/Camera.h"
#include "Device/Common.h"
#include "NormalLaunchParams.h"
#include "UniUtils/ConversionUtils.h"

using namespace EasyRender;
using namespace EasyRender::Programs::Normal;

extern "C" __constant__ Programs::Normal::LaunchParams param;

enum
{
    RADIANCE_TYPE = 0,
    RAY_TYPE_COUNT
};

extern "C" __global__ void __raygen__RenderFrame()
{
    auto idx_x = optixGetLaunchIndex().x, idx_y = optixGetLaunchIndex().y;
    auto idx = (std::size_t)optixGetLaunchDimensions().x * idx_y + idx_x;

    glm::vec3 result{ 0, 0, 0 };
    std::uint32_t u0, u1;
    PackPointer(&result, u0, u1);

    // Normally we need a scale to shift the ray direction, here just omit it.
    glm::vec3 rayDir =
        PinholeGenerateRay({ idx_x, idx_y }, param.fbSize, param.camera);

    optixTrace(param.traversable, UniUtils::ToFloat3(param.camera.pos),
               UniUtils::ToFloat3(rayDir), 1e-5, 1e30, 0, 255,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, RADIANCE_TYPE, RAY_TYPE_COUNT,
               RADIANCE_TYPE, u0, u1);

    result = result * 255.f;
    param.colorBuffer[idx] = glm::u8vec4{ result.x, result.y, result.z, 0xFF };
}

extern "C" __global__ void __miss__radiance()
{
    auto *result = GetPRD<glm::vec3>();
    *result = reinterpret_cast<MissData *>(optixGetSbtDataPointer())->bg_color;
}

extern "C" __global__ void __closesthit__radiance()
{
    auto *result = GetPRD<glm::vec3>();
    HitData *sbt = reinterpret_cast<HitData *>(optixGetSbtDataPointer());
    glm::ivec3 primIdx = sbt->indices[optixGetPrimitiveIndex()];
    glm::vec3 N = BarycentricByIndices(sbt->normals, primIdx,
                                             optixGetTriangleBarycentrics());
    *result = NormalToColor(N);
}

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */
}