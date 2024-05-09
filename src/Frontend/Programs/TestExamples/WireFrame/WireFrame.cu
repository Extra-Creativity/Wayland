#include <glm/glm.hpp>
#include <optix_device.h>

#include "Device/Camera.h"
#include "Device/Common.h"
#include "WireFrameLaunchParams.h"

using namespace EasyRender;
using namespace EasyRender::Device;

extern "C" __constant__ Programs::WireFrame::LaunchParams param;

struct Payload
{
    glm::vec3 radiance;
    glm::vec3 rayPos;
    glm::vec3 rayDir;
    unsigned int depth;
    bool done;
};

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

    std::uint32_t u0, u1;
    PackPointer(&prd, u0, u1);
    unsigned seed = tea<4>(idx, param.frameID);
    prd.rayDir =
        PinholeGenerateRay({ idx_x, idx_y }, param.fbSize, param.camera, seed);
    prd.depth = 0;
    prd.rayPos = param.camera.pos;
    prd.radiance = {};
    prd.done = false;

    do
    {
        optixTrace(param.traversable, UniUtils::ToFloat3(prd.rayPos),
                   UniUtils::ToFloat3(prd.rayDir), 1e-5, 1e30, 0, 255,
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT, RADIANCE_TYPE, RAY_TYPE_COUNT,
                   RADIANCE_TYPE, u0, u1);
    } while (prd.depth < 1000 && !prd.done);


    int frameID = param.frameID;
    glm::dvec4 thisFrame = { prd.radiance.x, prd.radiance.y, prd.radiance.z, 1.0f };
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

extern "C" __global__ void __miss__radiance()
{
    auto *prd = GetPRD<Payload>();
    prd->radiance = { 0.0, 0.0, 0.0 };
    prd->done = true;
}

extern "C" __global__ void __closesthit__radiance()
{
    auto *prd = GetPRD<Payload>();
    float2 w = optixGetTriangleBarycentrics();
    float e = 0.03;
    if (w.x < e || w.x > 1 - e || w.y < e || w.y > 1 - e)
    {
        prd->radiance = { 1, 0, 0 };
        prd->done = true;
    }
    else
    {
        auto hitPos = GetHitPosition();
        prd->rayPos = hitPos;
        prd->depth += 1;
    }
}

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */
}