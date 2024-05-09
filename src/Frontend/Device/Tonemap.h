#pragma once
#include <optix_device.h>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

namespace EasyRender::Device
{

__host__ __device__ __forceinline__ float Luminance(glm::vec3 rad)
{
    return glm::dot(rad, glm::vec3{ 0.212656, 0.715158, 0.072186 });
}

__host__ __device__ __forceinline__ glm::vec3 Lin2RGB(glm::vec3 rad)
{
    float R = 3.2406 * rad.x - 1.5372 * rad.y - 0.4986 * rad.z;
    float G = -0.9689 * rad.x + 1.87589 * rad.y + 0.0415 * rad.z;
    float B = 0.0557 * rad.x - 0.204 * rad.y + 1.057 * rad.z;
    return glm::vec3{ R, G, B };
}

__host__ __device__ __forceinline__ glm::vec3 Reinhard(glm::vec3 rad) {
    return rad / (rad + glm::vec3(1.0f));
}

__host__ __device__ __forceinline__ glm::vec3 LuminanceReinhard(
    glm::vec3 rad)
{
    float l_in = Luminance(rad);
    l_in = l_in < 0 ? 0 : l_in;
    return glm::clamp(rad / glm::vec3{ 1.f + l_in }, 0.f, 1.f);
}

__host__ __device__ __forceinline__ glm::vec3 AceApprox(glm::vec3 rad)
{
    rad *= 0.6f;
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return glm::clamp((rad * (a * rad + b)) / (rad * (c * rad + d) + e), 0.0f, 1.0f);
}

}