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

struct Payload
{
    glm::vec3 radiance;
    glm::vec3 rayPos;
    glm::vec3 rayDir;
    unsigned int depth;
    unsigned int seed;
    bool done;
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

    //if (idx == 0 && param.frameID==0) {
    //    param.areaLights[0].print();
    //}

    /* Generate random seed */
    prd.seed = tea<4>(idx, param.frameID);
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
                   UniUtils::ToFloat3(prd.rayDir), 1e-5, 1e30, 0, 255,
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT, RADIANCE_TYPE, RAY_TYPE_COUNT,
                   RADIANCE_TYPE, u0, u1);
        if (prd.done) {
            break;
        }
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

    glm::dvec4 thisFrame = { prd.radiance.x, prd.radiance.y, prd.radiance.z,
                            0xFF };
    int frameID = param.frameID;
    if (frameID == 0)
        param.radianceBuffer[idx] = thisFrame;
    else
    {
        glm::dvec4 lastFrame = param.radianceBuffer[idx];
        param.radianceBuffer[idx] = lastFrame * double(frameID / (frameID + 1.0f)) +
                                    thisFrame * double(1.0f / (frameID + 1.0f));
    }
    param.colorBuffer[idx] =
        glm::clamp(param.radianceBuffer[idx], 0.f, 1.f) * 255.0f;
}

extern "C" __global__ void __miss__radiance()
{
    auto *prd = GetPRD<Payload>();
    prd->radiance *=
        reinterpret_cast<MissData *>(optixGetSbtDataPointer())->bg_color;
    prd->done = true;
}

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
        prd->radiance *= mat->L;
        if (!mat->twoSided && glm::dot(N, prd->rayDir)>0)
            prd->radiance = { 0, 0, 0 };
        prd->done = true;
        return;
    }
    if (prd->depth >= 25)
    {
        prd->radiance = { 0, 0, 0 };
        prd->done = true;
        return;
    }
    if (glm::dot(N, prd->rayDir) > 0)
        N = -N;

    glm::vec3 hitPos = GetHitPosition();
    glm::vec3 rayDir = RandomSampleDir(N, prd->seed);

     auto cosWeight = fmaxf(0.f, glm::dot(rayDir, N));
     prd->radiance = prd->radiance* mat->Kd * cosWeight *
                     2.f; // pdf = 1 / 2pi, albedo = kd / pi
     prd->rayPos = hitPos;
     prd->rayDir = rayDir;

    return;
}

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */
}