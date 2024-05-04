#include "UniUtils/ConversionUtils.h"
#include "glm/glm.hpp"
#include <optix_device.h>

#include "Device/Common.h"
#include "Device/Pdf.h"
#include "Device/Sample.h"
#include "Device/Scene.h"

#include "AOLaunchParams.h"

using namespace EasyRender;
using namespace EasyRender::Device;
using namespace EasyRender::Programs::AO;

__constant__ float EPSILON = 1e-3;

extern "C" __constant__ Programs::AO::LaunchParams param;

/* PG id - 0 */
extern "C" __global__ void __raygen__RenderFrame()
{
    auto idx_x = optixGetLaunchIndex().x, idx_y = optixGetLaunchIndex().y;
    auto idx = (std::size_t)optixGetLaunchDimensions().x * idx_y + idx_x;
    Payload prd;

    /* Generate random seed */
    prd.seed = tea<4>(idx, param.frameID);

    std::uint32_t u0, u1;
    PackPointer(&prd, u0, u1);

    glm::dvec4 thisFrame = { 0, 0, 0, 0xFF };

    int pixelSampleCount = 4;
    for (int i = 0; i < pixelSampleCount; ++i)
    {
        prd.radiance = { 0, 0, 0 };
        prd.rayDir = PinholeGenerateRay({ idx_x, idx_y }, param.fbSize,
                                        param.camera, prd.seed);
        optixTrace(param.traversable, UniUtils::ToFloat3(param.camera.pos),
                   UniUtils::ToFloat3(prd.rayDir), EPSILON, 1e30, 0,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   RADIANCE_TYPE, RAY_TYPE_COUNT, RADIANCE_TYPE, u0, u1);

        int occlusionSampleCount = 32;
        float occlusionRadius = 1.f;
        for (int j = 0; j < occlusionSampleCount; ++j)
        {
            float pdf;
            glm::vec3 rayDir = SampleCosineHemisphere(prd.N, pdf, prd.seed);
            std::uint32_t vis = 0;
            optixTrace(param.traversable, UniUtils::ToFloat3(prd.hitPos),
                       UniUtils::ToFloat3(rayDir), EPSILON, occlusionRadius, 0,
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
                prd.radiance +=
                    glm::dot(rayDir, prd.N) / PI / occlusionSampleCount / pdf;
            }
        }

        thisFrame.x += prd.radiance.x / pixelSampleCount;
        thisFrame.y += prd.radiance.y / pixelSampleCount;
        thisFrame.z += prd.radiance.z / pixelSampleCount;
    }

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
    prd->radiance = { 0.f, 0.f, 0.f };
}

/* PG id - 2 */
extern "C" __global__ void __miss__shadow()
{
    /* Visibility = true */
    optixSetPayload_0(114514);
}

/* PG id - 3 */
extern "C" __global__ void __closesthit__radiance()
{
    auto *prd = GetPRD<Payload>();
    HitData *mat = reinterpret_cast<HitData *>(optixGetSbtDataPointer());
    glm::ivec3 indices = mat->indices[optixGetPrimitiveIndex()];
    glm::vec3 hitPos = GetHitPosition();
    glm::vec3 N = BarycentricByIndices(mat->normals, indices,
                                       optixGetTriangleBarycentrics());
    N = glm::normalize(N);
    if (glm::dot(N, prd->rayDir) > 0)
        N = -N;
    
    prd->N = N;
    prd->hitPos = hitPos;
}

/* PG id - 4 */
extern "C" __global__ void __anyhit__shadow() {}