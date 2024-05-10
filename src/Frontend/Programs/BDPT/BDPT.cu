#include "UniUtils/ConversionUtils.h"
#include "glm/glm.hpp"
#include <optix_device.h>

#include "Device/Common.h"
#include "Device/Pdf.h"
#include "Device/Sample.h"
#include "Device/Scene.h"
#include "Device/Material.h" 
#include "Device/Evaluate.h"
#include "Device/Tonemap.h"

#include "BDPTLaunchParams.h"

using namespace EasyRender;
using namespace EasyRender::Device;
using namespace EasyRender::Programs::BDPT;
 
__constant__ float EPSILON = 1e-3;

enum STRATEGY
{
    UPT = 0,
    NEE = 1,
    MIS = 2,
    STRATEGY_MAX
};

__constant__ STRATEGY strategy = MIS;

static __forceinline__ __device__ float UptMisWeight(float uptPdf, float neePdf)
{
    //assert(uptPdf + neePdf > 0);
    return uptPdf / (uptPdf + neePdf);
}

extern "C" __constant__ Programs::BDPT::LaunchParams param;

/* PG id - 0 */
extern "C" __global__ void __raygen__RenderFrame()
{
    auto idx_x = optixGetLaunchIndex().x, idx_y = optixGetLaunchIndex().y;
    auto idx = (std::size_t)optixGetLaunchDimensions().x * idx_y + idx_x;
    Payload prd;

    /* Generate random seed */
    prd.seed = tea<4>(idx, param.frameID);

    glm::dvec4 thisFrame = { 0.f, 0.f, 0.f, 1.f };

    int pixelSampleCount = 1;
    for (int i = 0; i < pixelSampleCount; ++i)
    {

        prd.depth = 0;
        prd.done = false;
        prd.radiance = { 0, 0, 0 };
        prd.throughput = { 1.f, 1.f, 1.f };
        prd.lastTraceTerm = { 1.f, 1.f, 1.f };
        prd.lastPdf = 1.0f;
        prd.rayPos = param.camera.pos;
        prd.rayDir = PinholeGenerateRay({ idx_x, idx_y }, param.fbSize,
                                        param.camera, prd.seed);

        std::uint32_t u0, u1;
        PackPointer(&prd, u0, u1);

        float RR_rate = 0.9;
        while (true)
        {

            optixTrace(param.traversable, UniUtils::ToFloat3(prd.rayPos),
                       UniUtils::ToFloat3(prd.rayDir), EPSILON, 1e30, 0,
                       OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                       RADIANCE_TYPE, RAY_TYPE_COUNT, RADIANCE_TYPE, u0, u1);
            if (prd.done)
            {
                break;
            }

            if (strategy != UPT)
            {
                LightSample ls;
                SampleAreaLightPos(param.areaLightCount, param.areaLights, ls,
                                   prd.seed);

                /* Visibility test */
                glm::vec3 visRay = ls.pos - prd.rayPos;
                float dist = glm::length(visRay);
                visRay /= dist;
                std::uint32_t vis = 0;
                optixTrace(
                    param.traversable, UniUtils::ToFloat3(prd.rayPos),
                    UniUtils::ToFloat3(visRay), EPSILON, dist * (1 - EPSILON),
                    0, OptixVisibilityMask(255),
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
                    float cosTheta2 = glm::dot(prd.lastNs, visRay);
                    auto &lt = param.areaLights[ls.areaLightID];
                    if (lt.twoSided || cosTheta1 < 0)
                    {
                        float neePdf = ls.pdf * dist * dist / fabsf(cosTheta1);
                        float uptPdf = 0.f;
                        if (strategy != NEE)
                        {
                            uptPdf = PdfDisneyBSDF(*prd.lastMat, prd.lastNs,
                                                   prd.lastNg,
                                                   -prd.lastRayDir, visRay);
                            if (uptPdf < 0 || isnan(uptPdf))
                                uptPdf = 0.f;
                        }
                        glm::vec3 BSDFVal = EvalDisneyBSDF(
                            *prd.lastMat, prd.lastNs, prd.lastNg,
                            -prd.lastRayDir, visRay, prd.lastTexcolor);
                        BSDFVal = clamp(BSDFVal, 0.f, 1e30f);
                        prd.radiance += prd.throughput * BSDFVal * fabsf(cosTheta2) *
                                        lt.L / neePdf *
                                        (1 - UptMisWeight(uptPdf, neePdf));
                    }
                }
            }
            if (prd.depth > 10)
            {
                if (rnd(prd.seed) > RR_rate)
                {
                    prd.throughput = { 0, 0, 0 };
                    break;
                }
                prd.throughput /= RR_rate;
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
    //param.colorBuffer[idx] =
        glm::clamp(param.radianceBuffer[idx], 0.f, 1.f) * 255.0f;
    glm::vec3 rad = glm::clamp(param.radianceBuffer[idx], 0.f, 1e30f);
    param.colorBuffer[idx] = glm::vec4{ AceApprox(rad), 1.0f } * 255.0f;
    //param.colorBuffer[idx] = glm::vec4{ Reinhard(rad), 1.0f } * 255.0f;
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
    /* Visibility = true */
    optixSetPayload_0(114514);
}

/* PG id - 3 */
extern "C" __global__ void __closesthit__radiance()
{
    auto *prd = GetPRD<Payload>();
    HitData *mat = reinterpret_cast<HitData *>(optixGetSbtDataPointer());
    prd->depth += 1;

    uint32_t primIdx = optixGetPrimitiveIndex();
    glm::ivec3 indices = mat->indices[optixGetPrimitiveIndex()];
    glm::vec3 Ns;
    glm::vec3 Ng;

    Ng = GetGeometryNormal(mat->vertices, indices);

    if (mat->hasNormal)
    {
        Ns = glm::normalize(BarycentricByIndices(
            mat->normals, indices, optixGetTriangleBarycentrics()));
    }
    else
    {
        Ns = Ng;
    }

    glm::vec3 hitPos = GetHitPosition();

    /* Hit light */
    if (mat->areaLightID < INVALID_INDEX)
    {
        auto &lt = param.areaLights[mat->areaLightID];
        if ((prd->depth == 1) ||
            (strategy != NEE && (lt.twoSided || glm::dot(Ns, prd->rayDir) < 0)))
        {
            /* Hit light directly */
            float neePdf = 0.f;
            if (strategy != UPT && prd->depth > 1)
            {
                LightSample ls;
                ls.pos = hitPos;
                ls.areaLightID = mat->areaLightID;
                PdfAreaLightPos(param.areaLightCount, param.areaLights, primIdx,
                                ls);
                float dist = optixGetRayTmax();
                neePdf =
                    ls.pdf * dist * dist / fabsf(glm::dot(Ns, prd->rayDir));
            }

            float uptPdf = prd->lastPdf;
            prd->radiance += prd->throughput * lt.L * prd->lastTraceTerm *
                             UptMisWeight(uptPdf, neePdf);
        }
        prd->done = true;
        return;
    }

    if (prd->depth >= 50)
    {
        prd->throughput = { 0, 0, 0 };
        prd->done = true;
        return;
    }

    if (mat->disneyMat.trans < 0.1)
    {
        if (glm::dot(Ng, prd->rayDir) > 0)
            Ng = -Ng;
        if (glm::dot(Ns, prd->rayDir) > 0)
            Ns = -Ns;
    }

    float pdf = 1.f;

    glm::vec3 rayDir =
        SampleDisneyBSDF(mat->disneyMat, Ns, Ng, -prd->rayDir, prd->seed);
    rayDir = glm::normalize(rayDir);
    pdf = PdfDisneyBSDF(mat->disneyMat, Ns, Ng, -prd->rayDir, rayDir);
    if (pdf < 0.0f || isnan(pdf)) {
        prd->done = true;
        return;
    }

    glm::vec3 texColor = { 1.f, 1.f, 1.f };
    if (mat->hasTexture)
    {
        glm::vec2 tc = BarycentricByIndices(mat->texcoords, indices,
                                            optixGetTriangleBarycentrics());
        texColor = UniUtils::ToVec4<glm::vec4>(
            tex2D<float4>(mat->texture, tc.x, tc.y));
    }
    /* Compute last BSDF */
    prd->throughput *= prd->lastTraceTerm;

    float cosWeight = glm::dot(rayDir, Ns);
    glm::vec3 BsdfTerm =
        EvalDisneyBSDF(mat->disneyMat, Ns, Ng, -prd->rayDir, rayDir, texColor);
    BsdfTerm = clamp(BsdfTerm, 0.f, 1e30f); 
    prd->lastTraceTerm = BsdfTerm * fabsf(cosWeight) / pdf;

    prd->rayPos = hitPos;
    prd->lastRayDir = prd->rayDir;
    prd->lastTexcolor = texColor;
    prd->rayDir = rayDir;
    prd->lastNs = Ns;
    prd->lastNg = Ng;
    prd->lastPdf = pdf;
    prd->lastMat = &mat->disneyMat;

    return;
}

/* PG id - 4 */
extern "C" __global__ void __anyhit__shadow() {}