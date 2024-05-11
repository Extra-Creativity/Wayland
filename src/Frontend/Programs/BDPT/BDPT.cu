#include "glm/glm.hpp"
#include "cuda_runtime.h"
#include <optix_device.h>

#include "Device/Common.h"
#include "Device/Pdf.h"
#include "Device/Sample.h"
#include "Device/Scene.h"
#include "Device/Material.h" 
#include "Device/Evaluate.h"
#include "Device/Tonemap.h"

#include "BDPTLaunchParams.h"
#include "BDPT.h"

using namespace EasyRender;
using namespace EasyRender::Device;
using namespace EasyRender::Programs::BDPT;
 
__constant__ float EPSILON = 1e-3;

extern "C" __constant__ Programs::BDPT::LaunchParams param;


__device__ __forceinline__ void InitPayload(Payload &prd)
{
    prd.depth = 0;
    prd.miss = false;
    prd.hitLight = false;
    prd.light = nullptr;
}

__device__ __forceinline__ int TraceEyeSubpath(BDPTVertex *eyePath, int maxSize, uint32_t &seed,
                    glm::ivec2 idx, glm::vec3 &radiance)
{
    Payload prd;
    InitPayload(prd);
    prd.seed = seed;
    prd.rayPos = param.camera.pos;
    glm::vec3 rayDir = PinholeGenerateRay(idx, param.fbSize, param.camera, prd.seed);

    std::uint32_t u0, u1;
    PackPointer(&prd, u0, u1);

    while (prd.depth < maxSize)
    {
        optixTrace(param.traversable, UniUtils::ToFloat3(prd.rayPos),
                   UniUtils::ToFloat3(rayDir), EPSILON, 1e30, 0,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   RADIANCE_TYPE, RAY_TYPE_COUNT, RADIANCE_TYPE, u0, u1);
        if (prd.miss)
        {
            break;
        }

        int vIdx = prd.depth - 1;
        BDPTVertex &e = eyePath[vIdx];


        e.pos = prd.rayPos;
        e.Ns = prd.Ns;
        e.Ng = prd.Ng;
        e.Wi = -rayDir;
        e.texcolor = prd.texcolor;
        e.mat = prd.mat;
        e.light = prd.light;
        e.depth = prd.depth;

        if (vIdx == 0)
            e.tput = glm::vec3{ 1.f,1.f,1.f };
        else {
            BDPTVertex &le = eyePath[vIdx - 1];
            glm::vec3 bsdfTerm = EvalDisneyBSDF(*le.mat, le.Ns, le.Ng, le.Wi,
                                                rayDir, le.texcolor);
            float cosWeight = glm::dot(rayDir, le.Ns);
            e.tput = le.tput * clamp(bsdfTerm, 0.f, 1e30f) * fabsf(cosWeight) / le.pdf;
        }

        if (prd.hitLight)
        {
            if (!prd.light->twoSided) {
                if (glm::dot(e.Ns, rayDir) > 0)
                    break;
            }
            if (vIdx == 0)
                radiance = prd.light->L;
            else
            {
                /* Compute radiance */
                radiance = prd.light->L * e.tput;
            }
            break;
        }


        /* Sample next direction */
        float samplePdf = 1.f;
        rayDir = SampleDisneyBSDF(*e.mat, e.Ns, e.Ng, e.Wi, prd.seed);
        rayDir = glm::normalize(rayDir);
        samplePdf = PdfDisneyBSDF(*e.mat, e.Ns, e.Ng, e.Wi, rayDir);
        /* Bad sample*/
        if (samplePdf < 0.0f || isnan(samplePdf))
            break;
        e.pdf = samplePdf;
    }

    seed = prd.seed;
    return prd.depth;
}


/* PG id - 0 */
extern "C" __global__ void __raygen__RenderFrame()
{
    auto idx_x = optixGetLaunchIndex().x, idx_y = optixGetLaunchIndex().y;
    auto idx = (std::size_t)optixGetLaunchDimensions().x * idx_y + idx_x;

    /* Generate random seed */
    uint32_t seed = tea<4>(idx, param.frameID);

    glm::vec3 thisFrame = { 0.f, 0.f, 0.f};

    int pixelSampleCount = 1;
    for (int i = 0; i < pixelSampleCount; ++i)
    {
        /* Trace Eye Subpath */
        BDPTVertex eyeSubPath[10];
        glm::vec3 radiance{0.f, 0.f, 0.f};
        int eyePathSize = TraceEyeSubpath(eyeSubPath, 10, seed,
                                          glm::ivec2{ idx_x, idx_y }, radiance);
        thisFrame += radiance / float(pixelSampleCount);
        if (eyePathSize == 0)
            continue;

        // BDPTVertex lightSubPath[10];
    }

    int frameID = param.frameID;
    if (frameID == 0)
        param.radianceBuffer[idx] = glm::vec4{ thisFrame,1.f};
    else
    {
        glm::vec4 lastFrame = param.radianceBuffer[idx];
        param.radianceBuffer[idx] =
            lastFrame * float(frameID / (frameID + 1.0f)) +
            glm::vec4{ thisFrame, 1.f } * float(1.0f / (frameID + 1.0f));
    }
    //param.colorBuffer[idx] =
    //    glm::clamp(param.radianceBuffer[idx], 0.f, 1.f) * 255.0f;
    glm::vec3 rad = glm::clamp(param.radianceBuffer[idx], 0.f, 1e30f);
    param.colorBuffer[idx] = glm::vec4{ AceApprox(rad), 1.0f } * 255.0f;
    //param.colorBuffer[idx] = glm::vec4{ Reinhard(rad), 1.0f } * 255.0f;
}

/* PG id - 1 */
extern "C" __global__ void __miss__radiance()
{
    auto *prd = GetPRD<Payload>();
    prd->miss = true;
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
        prd->hitLight = true;
        prd->light = &param.areaLights[mat->areaLightID];
    }
    else if (mat->disneyMat.trans < 0.1)
    {
        glm::vec3 rayDir =
            UniUtils::ToVec3<glm::vec3>(optixGetWorldRayDirection());
        if (glm::dot(Ng, rayDir) > 0)
            Ng = -Ng;
        if (glm::dot(Ns, rayDir) > 0)
            Ns = -Ns;
    }

    glm::vec3 texColor = { 1.f, 1.f, 1.f };
    if (mat->hasTexture)
    {
        glm::vec2 tc = BarycentricByIndices(mat->texcoords, indices,
                                            optixGetTriangleBarycentrics());
        texColor = UniUtils::ToVec4<glm::vec4>(
            tex2D<float4>(mat->texture, tc.x, tc.y));
    }

    prd->rayPos = hitPos;
    prd->texcolor = texColor;
    prd->Ns = Ns;
    prd->Ng = Ng;
    prd->mat = &mat->disneyMat;
    return;
}

/* PG id - 4 */
extern "C" __global__ void __anyhit__shadow() {}