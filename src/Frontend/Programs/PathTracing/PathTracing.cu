#include "Device/Camera.h"
#include "Device/Common.h"
#include "Device/Sample.h"
#include "DeviceUtils/Payload.h"
#include "UniUtils/ConversionUtils.h"
#include "glm/glm.hpp"
#include <optix_device.h>

#include "PathTracingLaunchParams.h"

using namespace EasyRender;
using namespace EasyRender::Device;
using namespace EasyRender::Programs::PathTracing;

__constant__ int minDepth = 5;
__constant__ float continuePossiblity = 0.99;
__constant__ float epsilon = 1e-5;

__constant__ bool UPT_only = 1;
__constant__ bool NEE_only = 0;
__constant__ bool PT_MIS = 0;

extern "C" __constant__ Programs::PathTracing::LaunchParams param;

/* PG id - 0 */
extern "C" __global__ void __raygen__RenderFrame()
{

    auto idx_x = optixGetLaunchIndex().x, idx_y = optixGetLaunchIndex().y;
    auto idx = (std::size_t)optixGetLaunchDimensions().x * idx_y + idx_x;
    Payload prd;

    /* Check rendering option */
    assert(int(UPT_only) + int(NEE_only) + int(PT_MIS) == 1);

    /* Generate random seed */
    prd.seed = tea<4>(idx, param.frameID);
    prd.depth = 0;
    prd.done = false;
    prd.radiance = { 0, 0, 0 };
    prd.throughput = { 1, 1, 1 };
    prd.lastTraceTerm = 1.f;
    prd.rayPos = param.camera.pos;
    prd.rayDir = PinholeGenerateRay({ idx_x, idx_y }, param.fbSize,
                                    param.camera, prd.seed);

    std::uint32_t u0, u1;
    PackPointer(&prd, u0, u1);

    float RR_rate = 0.8;
    while (true)
    {

        optixTrace(param.traversable, UniUtils::ToFloat3(prd.rayPos),
                   UniUtils::ToFloat3(prd.rayDir), 1e-3, 1e30, 0,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   RADIANCE_TYPE, RAY_TYPE_COUNT, RADIANCE_TYPE, u0, u1);
        if (prd.done)
        {
            break;
        }

        if (NEE_only || PT_MIS)
        {
            LightSample ls;
            SampleAreaLightPos(param.areaLightCount, param.areaLights, ls,
                               prd.seed);

            /* Visibility test */
            glm::vec3 visRay = ls.pos - prd.rayPos;
            float dist = glm::length(visRay);
            visRay /= dist;
            std::uint32_t vis = 0;
            optixTrace(param.traversable, UniUtils::ToFloat3(prd.rayPos),
                       UniUtils::ToFloat3(visRay), 1e-3, dist * (1 - 1e-3), 0,
                       OptixVisibilityMask(255),
                       // For shadow rays: skip any/closest hit shaders and
                       // terminate on first intersection with anything. The
                       // miss shader is used to mark if the light was visible.
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                           OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                           OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                       SHADOW_TYPE, RAY_TYPE_COUNT, SHADOW_TYPE, vis);
            if (vis > 0)
            {
                float cosTheta1 = glm::dot(ls.N, visRay);
                float cosTheta2 = glm::dot(prd.lastNormal, visRay);
                if ((ls.twoSided || cosTheta1 < 0) && cosTheta2 > 0)
                {
                    float cosTheta = fabsf(cosTheta1 * cosTheta2);
                    prd.radiance += prd.throughput * ls.L * cosTheta * 2.0f /
                                    (dist * dist) / ls.pdf;
                }
            }
        }

        if (prd.depth > 5)
        {
            if (rnd(prd.seed) > RR_rate)
            {
                prd.throughput = { 0, 0, 0 };
                break;
            }
            prd.throughput /= RR_rate;
        }
    }

    glm::dvec4 thisFrame = { prd.radiance.x, prd.radiance.y, prd.radiance.z,
                             0xFF };
    int frameID = param.frameID;
    if (frameID == 0)
        param.radianceBuffer[idx] = thisFrame;
    else
    {
        glm::dvec4 lastFrame = param.radianceBuffer[idx];
        param.radianceBuffer[idx] =
            lastFrame * double(frameID / (frameID + 1.0f)) +
            thisFrame * double(1.0f / (frameID + 1.0f));
    }
    param.colorBuffer[idx] =
        glm::clamp(param.radianceBuffer[idx], 0.f, 1.f) * 255.0f;
}

/* PG id - 1 */
extern "C" __global__ void __miss__radiance()
{
    auto *prd = GetPRD<Payload>();

    prd->radiance +=
        prd->throughput *
        reinterpret_cast<MissData *>(optixGetSbtDataPointer())->bg_color;
    prd->done = true;
}

/* PG id - 2 */
extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(114514);
}

/* PG id - 3 */
extern "C" __global__ void __closesthit__radiance()
{
    auto *prd = GetPRD<Payload>();
    HitData *mat = reinterpret_cast<HitData *>(optixGetSbtDataPointer());
    prd->depth += 1;

    glm::ivec3 indices = mat->indices[optixGetPrimitiveIndex()];
    glm::vec3 N = BarycentricByIndices(mat->normals, indices,
                                       optixGetTriangleBarycentrics());
    N = glm::normalize(N);

    /* Hit light */
    if (mat->L.x + mat->L.y + mat->L.z > 0)
    {
        if ((prd->depth <= 1 || !NEE_only) &&
            (mat->twoSided || glm::dot(N, prd->rayDir) < 0))
            prd->radiance += prd->throughput * mat->L * prd->lastTraceTerm * 2.0f;
        prd->done = true;
        return;
    }
    if (prd->depth >= 25)
    {
        prd->throughput = { 0, 0, 0 };
        prd->done = true;
        return;
    }
    if (glm::dot(N, prd->rayDir) > 0)
        N = -N;

    glm::vec3 hitPos = GetHitPosition();
    float pdf;
    glm::vec3 rayDir = SampleUniformHemisphere(N, pdf, prd->seed);

    auto cosWeight = fmaxf(0.f, glm::dot(rayDir, N));
    prd->throughput *=
        mat->Kd / PI; // pdf = 1 / 2pi, albedo = kd / pi
    prd->throughput *= prd->lastTraceTerm;
    prd->lastTraceTerm = cosWeight / pdf;
    prd->rayPos = hitPos;
    prd->rayDir = rayDir;
    prd->lastNormal = N;

    return;
}

/* PG id - 4 */
extern "C" __global__ void __anyhit__shadow() {}