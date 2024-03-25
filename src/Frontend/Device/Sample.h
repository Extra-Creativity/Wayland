#pragma once
#include "Random.h"
#include "Utils/MathConstants.h"
#include "cuda_runtime.h"
#include "glm/glm.hpp"

namespace EasyRender
{

__device__ __forceinline__ void GetOrthoNormalBasis(glm::vec3 vec, glm::vec3 &u,
                                                    glm::vec3 &v)
{
    float x = vec.x, y = vec.y, z = vec.z;

    float sign = copysignf(1.0f, z);
    float a = -1.0f / (sign + z);
    float b = x * y * a;

    /* Get onb */
    u = glm::vec3{ 1.0f + sign * x * x * a, sign * b, -sign * x };
    v = glm::vec3{ b, sign + y * y * a, -y };
    // if (glm::dot(u, v) > 1e-5 || glm::dot(u, v) < -1e-5 ||
    //     glm::dot(vec, u) > 1e-5 || glm::dot(vec, u) < -1e-5 ||
    //     glm::dot(vec, v) > 1e-5 || glm::dot(vec, v) < -1e-5)
    //     printf("wrong onb!\n");
    //if (glm::length(u) < 1 - 1e-5 || glm::length(u) > 1 + 1e-5 ||
    //    glm::length(v) < 1 - 1e-5 || glm::length(v) > 1 + 1e-5)
    //    printf("wrong onb base length! %f %f\n", glm::length(u),
    //           glm::length(u));
    return;
}

__device__ __forceinline__ glm::vec3 RandomSampleDir(glm::vec3 normal,
                                                     unsigned int &seed)
{
    glm::vec3 x, y;
    GetOrthoNormalBasis(normal, x, y);

    auto angle = rnd(seed) * TWO_PI, h = rnd(seed);
    return (sinf(angle) * x + cosf(angle) * y) * sqrtf(1 - h * h) + h * normal;
}

} // namespace EasyRender