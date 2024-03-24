#include "Device/Common.h"
#include "Device/Camera.h"
#include "Device/Sample.h"
#include "DeviceUtils/Payload.h"
#include "UniUtils/ConversionUtils.h"
#include "glm/glm.hpp"
#include <optix_device.h>

#include "PathTracingLaunchParams.h"

using namespace EasyRender;
using namespace EasyRender::Programs::PathTracing;

struct Payload
{
    glm::vec3 radiance;
    glm::vec3 rayPos;
    glm::vec3 rayDir;
    unsigned int depth;
    unsigned int seed;
};

__constant__ int minDepth = 5;
__constant__ float continuePossiblity = 0.99;
__constant__ float epsilon = 1e-5;

extern "C" __constant__ Programs::PathTracing::LaunchParams param;

enum
{
    RADIANCE_TYPE = 0,
    RAY_TYPE_COUNT
};

extern "C" __global__ void __raygen__RenderFrame()
{
    auto idx_x = optixGetLaunchIndex().x, idx_y = optixGetLaunchIndex().y;
    auto idx = (std::size_t)optixGetLaunchDimensions().x * idx_y + idx_x;

    Payload prd;

    /* Generate random seed */
    prd.seed = tea<4>(idx, param.frameID);
    prd.radiance = { 1, 1, 1 };
    prd.rayPos = param.camera.pos;
    prd.rayDir =
        PinholeGenerateRay({ idx_x, idx_y }, param.fbSize, param.camera, prd.seed);

    std::uint32_t u0, u1;
    PackPointer(&prd, u0, u1);

    float RR_rate = 0.9;
    do
    {
        optixTrace(param.traversable, UniUtils::ToFloat3(prd.rayPos),
                   UniUtils::ToFloat3(prd.rayDir), 1e-5, 1e30, 0, 255,
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT, RADIANCE_TYPE, RAY_TYPE_COUNT,
                   RADIANCE_TYPE, u0, u1);
        if (rnd(prd.seed) > RR_rate)
            break;
        prd.radiance /= RR_rate;
    } while (prd.depth < 25);

    glm::vec3 result;
    result = glm::clamp(prd.radiance, 0.f, 1.f) * 255.0f;
    param.colorBuffer[idx] = glm::u8vec4{ result.x, result.y, result.z, 0xFF };
}

extern "C" __global__ void __miss__radiance()
{
    auto *prd = GetPRD<Payload>();
    prd->radiance *=
        reinterpret_cast<MissData *>(optixGetSbtDataPointer())->bg_color;
}

extern "C" __global__ void __closesthit__radiance()
{
    auto *prd = GetPRD<Payload>();
    HitData *mat = reinterpret_cast<HitData *>(optixGetSbtDataPointer());

    prd->depth += 1;

    /* Hit light */
    if (mat->L.x + mat->L.y + mat->L.z > 0)
    {
        prd->radiance *=  mat->L;
        return;
    }

    //auto indices = data.indices[optixGetPrimitiveIndex()];
    //auto weights = optixGetTriangleBarycentrics();
    //auto normal = glm::normalize(
        //UniUtils::BarycentricByIndices(data.normals, indices, weights));

    /*auto hitPosition =
        UniUtils::ToVec3<glm::vec3>(optixGetWorldRayOrigin()) +
        UniUtils::ToVec3<glm::vec3>(optixGetWorldRayDirection()) *
            optixGetRayTmax();*/
    //auto rayDir = RandomSampleDir(normal, seed);
    //auto cosWeight = max(0.f, glm::dot(rayDir, normal));
    //float coeff = 2; // pdf = 1 / 2pi, albedo = kd / pi

    //auto &result = DeviceUtils::UnpackPayloads<Payload>(buffer);
    //DeviceUtils::SetToPayload<0>(result.color * data.color * cosWeight * coeff *
    //                             RR_factor);

}

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */
}