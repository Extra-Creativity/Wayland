#include "Param.h"
#include "cuda_device_runtime_api.h"
#include "optix_device.h"

#include "UniUtils/ConversionUtils.h"

using namespace Wayland;
using namespace Wayland::Example::Phong;

extern "C" __constant__ LaunchParam param;

template<typename T>
__device__ void PackPointer(T &data, std::uint32_t &u0, std::uint32_t &u1)
{
    auto ptr = reinterpret_cast<std::uintptr_t>(&data);
    u1 = ptr, u0 = ptr >> 32;
}

template<typename T>
__device__ T &UnpackPointer(std::uint32_t u0, std::uint32_t u1)
{
    return *reinterpret_cast<T *>(std::uintptr_t{ u0 } << 32 | u1);
}

template<typename T>
__device__ __forceinline__ T Barycentric(T v1, T v2, T v3, float2 coord)
{
    return v1 * (1 - coord.x - coord.y) + v2 * coord.x + v3 * coord.y;
}

__device__ glm::vec4 SampleTex(cudaTextureObject_t texture, glm::vec2 texCoords)
{
    return UniUtils::ToVec4<glm::vec4>(
        tex2D<float4>(texture, texCoords.x, 1 - texCoords.y));
}

extern "C" __global__ void __raygen__RenderFrame()
{
    auto rowNum = optixGetLaunchIndex().x, colNum = optixGetLaunchIndex().y;
    // scale to [-1, 1], row goes vertically.
    float verticalPos = 1 - 2.f * rowNum / optixGetLaunchDimensions().x,
          horizontalPos = 2.f * colNum / optixGetLaunchDimensions().y - 1;

    float3 result;
    std::uint32_t u0, u1;
    PackPointer(result, u0, u1);

    // In camera view, x = [1,0,0], y = [0,0,1], z = [0,-1,0]
    glm::vec3 rayDir = glm::normalize(
        param.camera.gaze +
        horizontalPos * -glm::cross(param.camera.gaze, param.camera.up) *
            param.camera.horizontalRatio +
        verticalPos * param.camera.up * param.camera.verticalRatio);

    optixTrace(param.traversable, UniUtils::ToFloat3(param.camera.position),
               UniUtils::ToFloat3(rayDir), 0, 100, 0, 255,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, u0, u1);

    auto idx = (std::size_t)optixGetLaunchDimensions().x * rowNum + colNum;
    param.colorBuffer[idx].r = result.x;
    param.colorBuffer[idx].g = result.y;
    param.colorBuffer[idx].b = result.z;
    param.colorBuffer[idx].a = 0xFF;
}

extern "C" __global__ void __miss__Plain()
{
    auto &result =
        UnpackPointer<float3>(optixGetPayload_0(), optixGetPayload_1());
    result = UniUtils::ToFloat3(
        reinterpret_cast<MissData *>(optixGetSbtDataPointer())->bgColor *
        255.0f);
}

extern "C" __global__ void __closesthit__Phong()
{
    auto &result =
        UnpackPointer<float3>(optixGetPayload_0(), optixGetPayload_1());
    auto &data = *reinterpret_cast<HitData *>(optixGetSbtDataPointer());

    auto indices = data.indices[optixGetPrimitiveIndex()];
    auto weights = optixGetTriangleBarycentrics();

    auto texCoords =
        Barycentric(data.texCoords[indices[0]], data.texCoords[indices[1]],
                    data.texCoords[indices[2]], weights);

    auto texColor = SampleTex(data.diffuseTexture, texCoords);
    result = UniUtils::ToFloat3(texColor * 255.0f);
}
