#include "Param.h"
#include "cuda_device_runtime_api.h"
#include "optix.h"

#include "DeviceUtils/Payload.h"
#include "UniUtils/ConversionUtils.h"
#include "UniUtils/MathUtils.h"

using namespace Wayland;
using namespace Wayland::Example::Phong;

extern "C" __constant__ LaunchParam param;

__device__ glm::vec4 SampleTex(cudaTextureObject_t texture, glm::vec2 texCoords)
{
    return UniUtils::ToVec4<glm::vec4>(
        tex2D<float4>(texture, texCoords.x, texCoords.y));
}

extern "C" __global__ void __raygen__RenderFrame()
{
    auto rowNum = optixGetLaunchIndex().x, colNum = optixGetLaunchIndex().y;
    // scale to [-1, 1], row goes vertically.
    float verticalPos = 1 - 2.f * rowNum / optixGetLaunchDimensions().x,
          horizontalPos = 2.f * colNum / optixGetLaunchDimensions().y - 1;

    // In camera view, x = [1,0,0], y = [0,0,1], z = [0,-1,0]
    glm::vec3 rayDir = glm::normalize(
        param.camera.gaze +
        horizontalPos * -glm::cross(param.camera.gaze, param.camera.up) *
            param.camera.horizontalRatio +
        verticalPos * param.camera.up * param.camera.verticalRatio);

    auto buffer = DeviceUtils::PackPayloads<glm::vec3>();

    DeviceUtils::optixTraceUnpack(buffer, param.traversable,
                                  UniUtils::ToFloat3(param.camera.position),
                                  UniUtils::ToFloat3(rayDir), 0, 100, 0, 255,
                                  OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0);

    auto &result = DeviceUtils::UnpackPayloads<glm::vec3>(buffer);
    auto idx = (std::size_t)optixGetLaunchDimensions().x * rowNum + colNum;
    param.colorBuffer[idx] = glm::u8vec4{ result.x, result.y, result.z, 0xFF };
}

extern "C" __global__ void __miss__Plain()
{
    auto result =
        reinterpret_cast<MissData *>(optixGetSbtDataPointer())->bgColor *
        255.0f;

    DeviceUtils::SetToPayload<0>(result);
}

extern "C" __global__ void __closesthit__Phong()
{
    auto &data = *reinterpret_cast<HitData *>(optixGetSbtDataPointer());

    auto indices = data.indices[optixGetPrimitiveIndex()];
    auto weights = optixGetTriangleBarycentrics();

    auto texCoords =
        UniUtils::BarycentricByIndices(data.texCoords, indices, weights);
    auto texColor = SampleTex(data.diffuseTexture, texCoords);

    auto normal = glm::normalize(
        UniUtils::BarycentricByIndices(data.normals, indices, weights));

    auto hitPosition =
        UniUtils::ToVec3<glm::vec3>(optixGetWorldRayOrigin()) +
        UniUtils::ToVec3<glm::vec3>(optixGetWorldRayDirection()) *
            optixGetRayTmax();

    auto lightDir = param.lightPos - hitPosition;
    auto lightDistanceSqr = glm::dot(lightDir, lightDir);
    lightDir /= sqrtf(lightDistanceSqr);

    auto viewDir = glm::normalize(param.camera.position - hitPosition);
    auto halfDir = glm::normalize(lightDir + viewDir);

    auto coeff =
        powf(max(0.f, glm::dot(halfDir, normal)), data.sPow) * data.ks +
        max(0.f, glm::dot(lightDir, viewDir)) * data.kd;
    glm::vec3 bgColor{ 10.0f };

    glm::vec3 result = texColor;
    result *= 255.0f * (param.lightColor * coeff + data.ka * bgColor) /
              lightDistanceSqr;

    DeviceUtils::SetToPayload<0>(result);
}
