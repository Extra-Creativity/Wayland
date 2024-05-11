#include "UniUtils/ConversionUtils.h"
#include "glm/glm.hpp"
#include <optix_device.h>

#include "Device/Common.h"
#include "Device/Pdf.h"
#include "Device/Sample.h"
#include "Device/Scene.h"
#include "Device/Material.h"
#include "Device/Evaluate.h"

#include "RandomWalkLaunchParams.h"

using namespace EasyRender;
using namespace EasyRender::Device;
using namespace EasyRender::Programs::RandomWalk;

__constant__ float EPSILON = 1e-3;

extern "C" __constant__ Programs::RandomWalk::LaunchParams param;

/* PG id - 0 */
extern "C" __global__ void __raygen__RenderFrame()
{
    auto idx_x = optixGetLaunchIndex().x, idx_y = optixGetLaunchIndex().y;
    auto idx = (std::size_t)optixGetLaunchDimensions().x * idx_y + idx_x;
    Payload prd;

    /* Generate random seed */
    prd.seed = tea<4>(idx, param.frameID);

    glm::dvec4 thisFrame = { 0.f, 0.f, 0.f, 1.f };

    prd.depth = 0;
    prd.done = false;
    prd.radiance = { 1, 1, 1 };
    prd.rayPos = param.camera.pos;
    prd.rayDir = PinholeGenerateRay({ idx_x, idx_y }, param.fbSize,
                                    param.camera, prd.seed);

    std::uint32_t u0, u1;
    PackPointer(&prd, u0, u1);

    float RR_rate = 0.8;
    while (true)
    {
        optixTrace(param.traversable, UniUtils::ToFloat3(prd.rayPos),
                   UniUtils::ToFloat3(prd.rayDir), EPSILON, 1e30, 0,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   RADIANCE_TYPE, RAY_TYPE_COUNT, RADIANCE_TYPE, u0, u1);
        if (prd.done)
            break;

        if (prd.depth > 5)
        {
            if (rnd(prd.seed) > RR_rate)
            {
                prd.radiance = { 0, 0, 0 };
                break;
            }
            prd.radiance /= RR_rate;
        }
    }

    int frameID = param.frameID;
    thisFrame = glm::dvec4(prd.radiance, 1.0f);
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
    prd->radiance *=
        reinterpret_cast<MissData *>(optixGetSbtDataPointer())->bg_color;
    prd->done = true;
}

/* PG id - 2 */
extern "C" __global__ void __miss__shadow() {}

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
        if (lt.twoSided || glm::dot(Ns, prd->rayDir) < 0)
        {
            prd->radiance *= lt.L;
        }
        else
        {
            prd->radiance = { 0, 0, 0 };
        }
        prd->done = true;
        return;
    }
    if (prd->depth >= 25)
    {
        prd->radiance = { 0, 0, 0 };
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

    float pdf;
    glm::vec3 tmp = Ns;
    if (mat->disneyMat.trans < 0.1f && rnd(prd->seed) > 0.5f)
        tmp = -Ns;
    glm::vec3 rayDir = SampleUniformHemisphere(tmp, pdf, prd->seed);

    glm::vec3 texColor = { 1.f, 1.f, 1.f };
    if (mat->hasTexture)
    {
        glm::vec2 tc = BarycentricByIndices(mat->texcoords, indices,
                                            optixGetTriangleBarycentrics());
        texColor = UniUtils::ToVec4<glm::vec4>(
            tex2D<float4>(mat->texture, tc.x, tc.y));
    }

    auto cosWeight = fmaxf(0.f, glm::dot(rayDir, Ns));

    prd->radiance *=
        EvalDisneyBSDF(mat->disneyMat, Ns, Ng, -prd->rayDir, rayDir, texColor);
    prd->radiance *= cosWeight / pdf;
    prd->rayPos = hitPos;
    prd->rayDir = rayDir;
    return;
}

/* PG id - 4 */
extern "C" __global__ void __anyhit__shadow() {}