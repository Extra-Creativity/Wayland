//#include "../../src/Backend/OptiX/Utils/UniUtils/ConversionUtils.h"
//#include "Param.h"
#include <optix_device.h>

//extern "C" __constant__ LaunchParams param;

//using namespace Wayland;

//template<typename T>
//__device__ void PackPointer(T &data, std::uint32_t &u0, std::uint32_t &u1)
//{
//    auto ptr = reinterpret_cast<std::uintptr_t>(&data);
//    u1 = ptr, u0 = ptr >> 32;
//}
//
//template<typename T>
//__device__ T &UnpackPointer(std::uint32_t u0, std::uint32_t u1)
//{
//    return *reinterpret_cast<T *>(std::uintptr_t{ u0 } << 32 | u1);
//}

extern "C" __global__ void __raygen__RenderFrame()
{
    //auto rowNum = optixGetLaunchIndex().x, colNum = optixGetLaunchIndex().y;
    // scale to [-1, 1], row goes vertically.
    // float verticalPos = 2.f * rowNum / optixGetLaunchDimensions().x - 1,
    //       horizontalPos = 2.f * colNum / optixGetLaunchDimensions().y - 1;

    //float3 result{0.8,0.8,0.8};
    // std::uint32_t u0, u1;
    // PackPointer(result, u0, u1);

    // Normally we need a scale to shift the ray direction, here just omit it.
    // In camera view, x = [1,0,0], y = [0,0,1], z = [0,-1,0]
    // glm::vec3 rayDir = glm::normalize(
    //     param.camera.gaze +
    //     horizontalPos * -glm::cross(param.camera.gaze, param.camera.up) +
    //     verticalPos * param.camera.up);

    // optixTrace(param.traversable, UniUtils::ToFloat3(param.camera.position),
    //            UniUtils::ToFloat3(rayDir), 0, 5, 0, 255,
    //            OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, u0, u1);

    //auto idx = (std::size_t)optixGetLaunchDimensions().x * rowNum + colNum;
    //param.colorBuffer[idx].r = result.x;
    //param.colorBuffer[idx].g = result.y;
    //param.colorBuffer[idx].b = result.z;
    //param.colorBuffer[idx].a = 0xFF;
}

extern "C" __global__ void __closesthit__radiance()
{
    // auto &result =
    //     UnpackPointer<float3>(optixGetPayload_0(), optixGetPayload_1());
    // result = UniUtils::ToFloat3(
    //     reinterpret_cast<HitData *>(optixGetSbtDataPointer())->diffuse_color *
    //     255.0f);
}

extern "C" __global__ void __miss__radiance()
{
    // auto &result =
    //     UnpackPointer<float3>(optixGetPayload_0(), optixGetPayload_1());
    // result = UniUtils::ToFloat3(
    //     reinterpret_cast<MissData *>(optixGetSbtDataPointer())->bg_color * 255.0f);
}
