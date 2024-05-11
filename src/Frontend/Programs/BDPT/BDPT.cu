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
 
extern "C" __constant__ Programs::BDPT::LaunchParams param;

enum MISMode
{
    UniformHeuristic = 0,
    BalanceHeuristic,
    PowerHeuristic
};

__constant__ float EPSILON = 1e-3;
__constant__ bool USE_UPT = false;
__constant__ bool PT_MODE = false;
__constant__ MISMode MIS_MODE = UniformHeuristic;

__device__ __forceinline__ void InitPayload(Payload &prd)
{
    prd.depth = 0;
    prd.miss = false;
    prd.hitLight = false;
    prd.light = nullptr;
}

/* Compute pdf in area measure */ 
__device__ __forceinline__ float ComputePdfForward(glm::vec3 wi, BDPTVertex &mid,
                                                BDPTVertex &next)
{
    glm::vec3 d = next.pos - mid.pos;
    float dist = glm::length(d);
    d /= dist;
    float pdfBsdf = PdfDisneyBSDF(*mid.mat, mid.Ns, mid.Ng, wi, d);
    return clamp(pdfBsdf, 1e-30f, 1e30f) * fabs(glm::dot(next.Ns, d)) / (dist * dist);
}

__device__ __forceinline__ float ComputePdfLightDir(BDPTVertex &l, BDPTVertex &e)
{
    glm::vec3 d = e.pos - l.pos;
    float dist = glm::length(d);
    d /= dist;
    return PdfCosineHemisphere(d, l.Ns) * fabsf(glm::dot(d, e.Ns)) / (dist * dist);
}

/* BDPT mis weight */
__device__ __forceinline__ float MISweight(BDPTVertex *eyePath,
                BDPTVertex *lightPath, int eyeEnd,
                int lightEnd)
{
    int numStrategy = (PT_MODE ? 1 : (eyeEnd + lightEnd + 1)) + int(USE_UPT);
    if (!USE_UPT && lightEnd == -1)
        return 0.0f;

    if (PT_MODE && lightEnd != 0 && lightEnd != -1)
        return 0.0f;

    if (MIS_MODE == UniformHeuristic)
        return 1.0f / numStrategy;
    
    /**
     *              p_el     p_ll
     *  e1  ->  e2  --->  l2  ->  l1
     *  e1  <-  e2  <---  l2  <-  l1
     *      p_ee    p_le
     */
//    std::vector<Float> pdfForward;
//    std::vector<Float> pdfInverse;
//    Float p_ee, p_el, p_le, p_ll;
//
//    if (lightEnd == -1)
//    {
//        /* PT */
//        for (int i = 1; i <= eyeEnd; ++i)
//        {
//            pdfForward.push_back(eyePath[i]->pdf);
//            pdfInverse.push_back(eyePath[i]->pdfInverse);
//        }
//    }
//    else if (eyeEnd == 0 && lightEnd == 0)
//    {
//        /* One eye vertex and one light vertex */
//        BDPTVertex *e = eyePath[0];
//        BDPTVertex *l = lightPath[0];
//        p_el = computePdfForward(e->wi, e, l);
//        p_le = computePdfLightDir(l, e);
//        pdfForward.push_back(p_el);
//        pdfInverse.push_back(p_le);
//    }
//    else if (eyeEnd == 0)
//    {
//        /* One eye vertex */
//        BDPTVertex *e = eyePath[0];
//        BDPTVertex *l = lightPath[lightEnd];
//        BDPTVertex *ll = lightPath[lightEnd - 1];
//        Vector3 d = normalize(l->pos - e->pos);
//        p_el = computePdfForward(e->wi, e, l);
//        p_le = computePdfForward(l->wi, l, e);
//        p_ll = computePdfForward(-d, l, ll);
//
//        pdfForward.push_back(p_el);
//        pdfInverse.push_back(p_le);
//        pdfForward.push_back(p_ll);
//        pdfInverse.push_back(ll->pdfInverse);
//
//        for (int i = lightEnd - 2; i >= 0; --i)
//        {
//            pdfForward.push_back(lightPath[i]->pdf);
//            pdfInverse.push_back(lightPath[i]->pdfInverse);
//        }
//    }
//    else if (lightEnd == 0)
//    {
//        /* One light vertex */
//        BDPTVertex *ee = eyePath[eyeEnd - 1];
//        BDPTVertex *e = eyePath[eyeEnd];
//        BDPTVertex *l = lightPath[0];
//        Vector3 d = normalize(l->pos - e->pos);
//
//        p_ee = computePdfForward(d, e, ee);
//        p_el = computePdfForward(e->wi, e, l);
//        p_le = computePdfLightDir(l, e);
//
//        for (int i = 1; i < eyeEnd; ++i)
//        {
//            pdfForward.push_back(eyePath[i]->pdf);
//            pdfInverse.push_back(eyePath[i]->pdfInverse);
//        }
//        pdfForward.push_back(e->pdf);
//        pdfInverse.push_back(p_ee);
//        pdfForward.push_back(p_el);
//        pdfInverse.push_back(p_le);
//    }
//    else
//    {
//        /* Other case */
//        BDPTVertex *ee = eyePath[eyeEnd - 1];
//        BDPTVertex *e = eyePath[eyeEnd];
//        BDPTVertex *l = lightPath[lightEnd];
//        BDPTVertex *ll = lightPath[lightEnd - 1];
//        Vector3 d = normalize(l->pos - e->pos);
//
//        p_ee = computePdfForward(d, e, ee);
//        p_el = computePdfForward(e->wi, e, l);
//        p_le = computePdfForward(l->wi, l, e);
//        p_ll = computePdfForward(-d, l, ll);
//
//        for (int i = 1; i < eyeEnd; ++i)
//        {
//            pdfForward.push_back(eyePath[i]->pdf);
//            pdfInverse.push_back(eyePath[i]->pdfInverse);
//        }
//        pdfForward.push_back(e->pdf);
//        pdfInverse.push_back(p_ee);
//        pdfForward.push_back(p_el);
//        pdfInverse.push_back(p_le);
//        pdfForward.push_back(p_ll);
//        pdfInverse.push_back(ll->pdfInverse);
//        for (int i = lightEnd - 2; i >= 0; --i)
//        {
//            pdfForward.push_back(lightPath[i]->pdf);
//            pdfInverse.push_back(lightPath[i]->pdfInverse);
//        }
//    }
//    pdfInverse.push_back(lightPath[0]->pdfLight);

    int curStrategy = eyeEnd;
    float denominator = 1.0f;
//
//    if (m_MISmode == BalanceHeuristic)
//    {
//        Float tmp = 1.0f;
//        for (int i = curStrategy - 1; i >= 0; --i)
//        {
//            tmp *= pdfInverse[i + 1] / pdfForward[i];
//#ifdef ONLY_PT
//            if (i == numStrategy - 1 || i == numStrategy - 2)
//#endif
//                denominator += tmp;
//        }
//        tmp = 1.0f;
//        for (int i = curStrategy + 1; i < numStrategy; ++i)
//        {
//            tmp *= pdfForward[i - 1] / pdfInverse[i];
//#ifdef ONLY_PT
//            if (i == numStrategy - 1 || i == numStrategy - 2)
//#endif
//                denominator += tmp;
//        }
//    }
//    if (m_MISmode == PowerHeuristic)
//    {
//        Float tmp = 1.0f;
//        for (int i = curStrategy - 1; i >= 0; --i)
//        {
//            tmp *= pdfInverse[i + 1] / pdfForward[i];
//#ifdef ONLY_PT
//            if (i == numStrategy - 1 || i == numStrategy - 2)
//#endif
//                denominator += tmp * tmp;
//        }
//        tmp = 1.0f;
//        for (int i = curStrategy + 1; i < numStrategy; ++i)
//        {
//            tmp *= pdfForward[i - 1] / pdfInverse[i];
//#ifdef ONLY_PT
//            if (i == numStrategy - 1 || i == numStrategy - 2)
//#endif
//                denominator += tmp * tmp;
//        }
//    }
//
    float ans = 1.0f / denominator;

    if (ans < 0 || isnan(ans) || isinf(ans))
        return 0.0f;
    else
        return ans;
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
        {
            e.tput = glm::vec3{ 1.f, 1.f, 1.f };
            e.pdf = 1.f;
        }
        else {
            BDPTVertex &le = eyePath[vIdx - 1];
            glm::vec3 bsdfTerm = EvalDisneyBSDF(*le.mat, le.Ns, le.Ng, le.Wi,
                                                rayDir, le.texcolor);
            float cosWeight = glm::dot(rayDir, le.Ns);

            float pdf = PdfDisneyBSDF(*le.mat, le.Ns, le.Ng, le.Wi, rayDir);
            pdf = clamp(pdf, 1e-30f, 1e30f);
            e.tput =
                le.tput * clamp(bsdfTerm, 0.f, 1e30f) * fabsf(cosWeight) / pdf;

            glm::vec3 d = e.pos - le.pos;
            float dist2 = glm::dot(d, d);
            e.pdf = pdf * fabsf(glm::dot(e.Wi, e.Ns)) / dist2;
        }

        /* Compute pdfInverse */
        if (vIdx >= 2) {
            BDPTVertex &le = eyePath[vIdx - 1];
            BDPTVertex &lle = eyePath[vIdx - 2];
            le.pdfInverse = ComputePdfForward(-e.Wi, le, lle);
        }

        if (prd.hitLight)
        {
            if (!prd.light->twoSided) {
                if (glm::dot(e.Ns, rayDir) > 0)
                    break;
            }
            if (vIdx == 0)
                radiance = prd.light->L;
            else if (USE_UPT)
            {

                BDPTVertex &le = eyePath[vIdx - 1];

                e.pdfInverse = ComputePdfLightDir(e, le);
                e.pdfLight = PdfAreaLightPos(param.areaLightCount, *e.light, prd.primIdx);

                /* Compute radiance */
                radiance = prd.light->L * e.tput * MISweight(eyePath, &e,
                                                          prd.depth-1, -1);
            }
            break;
        }

        /* Sample next direction */
        rayDir = SampleDisneyBSDF(*e.mat, e.Ns, e.Ng, e.Wi, prd.seed);
        rayDir = glm::normalize(rayDir);
    }

    seed = prd.seed;
    if (prd.hitLight)
        prd.depth -= 1;
    return prd.depth;
}

__device__ __forceinline__ int TraceLightSubpath(BDPTVertex *lightPath,
                                                 int maxSize, uint32_t &seed)
{
    /* Sample light */
    LightSample ls;
    float lightDirPdf;
    SampleAreaLightPos(param.areaLightCount, param.areaLights, ls, seed);
    glm::vec3 rayDir = SampleCosineHemisphere(ls.N, lightDirPdf, seed);

    Payload prd;
    InitPayload(prd);
    prd.depth = 1;
    prd.seed = seed;
    prd.rayPos = ls.pos;

    BDPTVertex &l = lightPath[0];
    l.pos = ls.pos;
    l.Ns = ls.N;
    l.Ng = ls.N;
    l.texcolor = { 1.f, 1.f, 1.f };
    l.mat = nullptr;
    //l.pdf = ls.pdf;
    l.pdfLight = ls.pdf;
    l.light = &param.areaLights[ls.areaLightID];
    l.tput = l.light->L / ls.pdf;

    std::uint32_t u0, u1;
    PackPointer(&prd, u0, u1);
    while (prd.depth < maxSize)
    {
        optixTrace(param.traversable, UniUtils::ToFloat3(prd.rayPos),
                   UniUtils::ToFloat3(rayDir), EPSILON, 1e30, 0,
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                   RADIANCE_TYPE, RAY_TYPE_COUNT, RADIANCE_TYPE, u0, u1);
        if (prd.miss || prd.hitLight)
        {
            break;
        }

        int vIdx = prd.depth - 1;
        BDPTVertex &v = lightPath[vIdx];

        v.pos = prd.rayPos;
        v.Ns = prd.Ns;
        v.Ng = prd.Ng;
        v.Wi = -rayDir;
        v.texcolor = prd.texcolor;
        v.mat = prd.mat;
        v.light = nullptr;
        v.depth = prd.depth;

        BDPTVertex &lv = lightPath[vIdx - 1];
        /* pdf/pdfInverse is opposite comparing to eye trace! */
        /*  v <- lv <- llv */
        if (vIdx == 1)
        {
            glm::vec3 d = v.pos - lv.pos;
            float dist = glm::length(d);
            d /= dist;
            v.tput = lightPath[0].tput * fabsf((glm::dot(d, lv.Ns)))/ lightDirPdf;
            lv.pdfInverse = lightDirPdf * fabsf(glm::dot(rayDir, v.Ns)) / (dist * dist);
        }
        else
        {
            glm::vec3 bsdfTerm = EvalDisneyBSDF(*lv.mat, lv.Ns, lv.Ng, lv.Wi,
                                                rayDir, lv.texcolor);
            float cosWeight = glm::dot(rayDir, lv.Ns);

            float pdf = PdfDisneyBSDF(*lv.mat, lv.Ns, lv.Ng, lv.Wi, rayDir);
            pdf = clamp(pdf, 1e-30f, 1e30f);
            v.tput =
                lv.tput * clamp(bsdfTerm, 0.f, 1e30f) * fabsf(cosWeight) / pdf;
            glm::vec3 d = v.pos - lv.pos;
            lv.pdfInverse = pdf * fabsf(glm::dot(v.Wi, v.Ns)) / glm::dot(d, d);

            BDPTVertex &llv = lightPath[vIdx - 2];
            llv.pdf = ComputePdfForward(-v.Wi, lv, llv);
        }

        /* Sample next direction */
        rayDir = SampleDisneyBSDF(*v.mat, v.Ns, v.Ng, v.Wi, prd.seed);
        rayDir = glm::normalize(rayDir);
    }

    seed = prd.seed;
    if (prd.hitLight)
        prd.depth -= 1;
    return prd.depth;
}


__device__ __forceinline__ glm::vec3 EvalContri(BDPTVertex &eyeEnd,
                                                BDPTVertex &lightEnd)
{
    glm::vec3 visRay = lightEnd.pos - eyeEnd.pos;;
    float dist = glm::length(visRay);
    visRay /= dist;

    std::uint32_t vis = 0;
    /* Shadow test */
    optixTrace(param.traversable, UniUtils::ToFloat3(eyeEnd.pos),
               UniUtils::ToFloat3(visRay), EPSILON, dist * (1 - EPSILON), 0,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                   OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               SHADOW_TYPE, RAY_TYPE_COUNT, SHADOW_TYPE, vis);

    glm::vec3 contri{ 0.f, 0.f, 0.f };
    
    if (!vis)
        return contri;
    if (lightEnd.light && !lightEnd.light->twoSided &&
        glm::dot(lightEnd.Ns, visRay) > 0)
        return contri;

    glm::vec3 bsdfEye = EvalDisneyBSDF(*eyeEnd.mat, eyeEnd.Ns, eyeEnd.Ng,
                                       eyeEnd.Wi, visRay, eyeEnd.texcolor);

    contri = eyeEnd.tput * bsdfEye * fabsf(glm::dot(eyeEnd.Ns, visRay)) *
             fabsf(glm::dot(lightEnd.Ns, visRay)) * lightEnd.tput /
             (dist * dist);

    if (lightEnd.light == nullptr)
    {
        contri *= EvalDisneyBSDF(*lightEnd.mat, lightEnd.Ns, lightEnd.Ng,
                                 -visRay, lightEnd.Wi, lightEnd.texcolor);
    }
    return contri;
}

__device__ __forceinline__ glm::vec3 ConnectSubpaths(BDPTVertex *eyePath,
                                                     int eyeSize,
                                                     BDPTVertex *lightPath,
                                                     int lightSize)
{
    if (eyeSize == 0 || lightSize == 0)
        return;
    glm::vec3 radiance = { 0.f, 0.f, 0.f };
    for (int i = 0; i < eyeSize; ++i)
    {
        for (int j = 0; j < lightSize; ++j)
        {
            BDPTVertex &eyeEnd = eyePath[i];
            BDPTVertex &lightEnd = lightPath[j];
            glm::vec3 contri = EvalContri(eyeEnd, lightEnd);
            if (!(contri.x + contri.y + contri.z > 1e-10f))
                continue;
            radiance += contri * MISweight(eyePath, lightPath, i, j);
        }
    }
    return radiance;
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

        BDPTVertex lightSubPath[10];
        int lightPathSize = TraceLightSubpath(lightSubPath, 10, seed);
        
        thisFrame += ConnectSubpaths(eyeSubPath, eyePathSize, lightSubPath,
                                     lightPathSize) / float(pixelSampleCount);
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
    prd->primIdx = primIdx;
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
    prd->hitLight = false;
    prd->light = nullptr;
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