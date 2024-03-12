#include "Param.h"
#include "Random.h"
#include "cuda_device_runtime_api.h"

#include "DeviceUtils/Payload.h"
#include "UniUtils/ConversionUtils.h"
#include "UniUtils/MathUtils.h"

using namespace Wayland;
using namespace Wayland::Example::PathTracing;

struct Payload
{
    glm::vec3 color;
    unsigned int depth;
    unsigned int seed;
};

extern "C" __constant__ LaunchParam param;
__constant__ int minDepth = 5;
__constant__ float stopPossiblity = 0.99;

__device__ __forceinline__ void GetOrthoNormalBasis(glm::vec3 vec, glm::vec3 &u,
                                                    glm::vec3 &v)
{
    float x = vec.x, y = vec.y, z = vec.z;

    float sign = copysignf(1.0f, z);
    float a = -1.0f / (sign + z);
    float b = x * y * a;

    u = glm::vec3{ 1.0f + sign * x * x * a, sign * b, -sign * x };
    v = glm::vec3{ b, sign + y * y * a, -y };
    return;
}

__device__ __forceinline__ glm::vec3 RandomSampleDir(glm::vec3 normal,
                                                     unsigned int &seed)
{
    glm::vec3 x, y;
    GetOrthoNormalBasis(normal, x, y);

    auto angle = rnd(seed) * 2 * 3.141592653589793f, radius = rnd(seed);
    return (sinf(angle) * x + cosf(angle) * y) * radius +
           sqrtf(1 - radius * radius) * normal;
}

extern "C" __global__ void __raygen__RenderFrame()
{
    auto rowNum = optixGetLaunchIndex().x, colNum = optixGetLaunchIndex().y;
    glm::vec2 cellSize{ 1.f / optixGetLaunchDimensions().x,
                        1.f / optixGetLaunchDimensions().y };
    // scale to [-1, 1], row goes vertically.
    float verticalPos = 1 - 2.f * rowNum * cellSize.x,
          horizontalPos = 2.f * colNum * cellSize.y - 1;

    glm::vec3 horizontalVec = glm::cross(param.camera.gaze, param.camera.up) *
                              param.camera.horizontalRatio,
              verticalVec = param.camera.up * param.camera.verticalRatio;

    // In camera view, x = [-1,0,0], y = [0,1,0], z = [0,0,-1]
    glm::vec3 rayDir =
        glm::normalize(param.camera.gaze + horizontalPos * horizontalVec +
                       verticalPos * verticalVec);

    auto idx = (std::size_t)optixGetLaunchDimensions().x * rowNum + colNum;

    // Pass payload to the closest hit.
    auto buffer = DeviceUtils::PackPayloads(
        Payload{ glm::vec3{}, param.maxDepth, tea<4>(idx, 10086) });
    glm::vec3 result{ 0 };
    float sampleWeight = 1.f / param.sampleNum;
    for (unsigned int i = 0; i < param.sampleNum; i++)
    {
        auto &payload = DeviceUtils::UnpackPayloads<Payload>(buffer);
        glm::vec3 randomRayDir = // random position in a pixel
            rayDir + rnd(payload.seed) * 2 * cellSize.y * horizontalVec +
            rnd(payload.seed) * 2 * cellSize.x * verticalVec;
        DeviceUtils::optixTraceUnpack(buffer, param.traversable,
                                      UniUtils::ToFloat3(param.camera.position),
                                      UniUtils::ToFloat3(randomRayDir), 0, 30,
                                      0, 255, OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0,
                                      1, 0);
        result += payload.color * sampleWeight;
    }
    result = glm::clamp(result, 0.f, 1.f) * 255.0f;
    param.colorBuffer[idx] = glm::u8vec4{ result.x, result.y, result.z, 0xFF };
}

extern "C" __global__ void __miss__Empty()
{
    // If miss, make is ambient i.e. zero.
    DeviceUtils::SetToPayload<0>(glm::vec3{ 0 });
}

extern "C" __global__ void __closesthit__PT()
{
    auto &data = *reinterpret_cast<HitData *>(optixGetSbtDataPointer());
    auto currDepth = DeviceUtils::GetFromPayload<unsigned int, 3>();
    auto seed = DeviceUtils::GetFromPayload<unsigned int, 4>();

    if (currDepth == 0 || data.emission != glm::vec3{ 0, 0, 0 })
    {
        // Return emission back.
        DeviceUtils::SetToPayload<0>(data.emission);
        return;
    }
    else if (currDepth <= minDepth)
    {
        if (rnd(seed) > stopPossiblity)
        {
            DeviceUtils::SetToPayload<0>(data.emission);
            DeviceUtils::SetToPayload<4>(seed);
            return;
        }
    }

    auto indices = data.indices[optixGetPrimitiveIndex()];
    auto weights = optixGetTriangleBarycentrics();
    auto normal = glm::normalize(
        UniUtils::BarycentricByIndices(data.normals, indices, weights));

    auto hitPosition =
        UniUtils::ToVec3<glm::vec3>(optixGetWorldRayOrigin()) +
        UniUtils::ToVec3<glm::vec3>(optixGetWorldRayDirection()) *
            optixGetRayTmax();
    auto rayDir = RandomSampleDir(normal, seed);
    auto cosWeight = max(0.f, glm::dot(rayDir, normal));
    float coeff = 2; // pdf = 1 / 2pi, albedo = kd / pi

    // continue to trace with new payload.
    auto buffer =
        DeviceUtils::PackPayloads(Payload{ glm::vec3{}, currDepth - 1, seed });
    DeviceUtils::optixTraceUnpack(buffer, param.traversable,
                                  UniUtils::ToFloat3(hitPosition),
                                  UniUtils::ToFloat3(rayDir), 0, 30, 0, 20,
                                  OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0);
    auto &result = DeviceUtils::UnpackPayloads<Payload>(buffer);
    DeviceUtils::SetToPayload<0>(result.color * data.color * cosWeight * coeff /
                                 stopPossiblity);
    DeviceUtils::SetToPayload<4>(result.seed);
}
