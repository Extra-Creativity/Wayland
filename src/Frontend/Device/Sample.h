#pragma once
#include "cuda_runtime.h"
#include "glm/glm.hpp"
#include "Random.h"
#include "Utils/MathConstants.h"

namespace EasyRender
{

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

    auto angle = rnd(seed) * 2 * TWO_PI, h = rnd(seed);
    return (sinf(angle) * x + cosf(angle) * y) * sqrtf(1 - h * h) + h * normal;
}

} // namespace EasyRender