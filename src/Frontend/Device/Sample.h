#pragma once
#include "Common.h"
#include "Light.h"
#include "Random.h"
#include "Utils/MathConstants.h"
#include "cuda_runtime.h"
#include "glm/glm.hpp"

namespace EasyRender::Device
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
    return;
}

/* Uniformly samples a hemisphere */
__device__ __forceinline__ glm::vec3 SampleUniformHemisphere(glm::vec3 normal,
                                                             float &pdf,
                                                             unsigned int &seed)
{
    glm::vec3 x, y;
    GetOrthoNormalBasis(normal, x, y);

    auto angle = rnd(seed) * TWO_PI, h = rnd(seed);
    pdf = 1.0f / TWO_PI;
    return (sinf(angle) * x + cosf(angle) * y) * sqrtf(1 - h * h) + h * normal;
}

/* The correctness of this function needs tests */
__device__ __forceinline__ glm::vec3 SampleTrianglePos(glm::vec3 *vertices,
                                                       glm::ivec3 indices,
                                                       unsigned int &seed)
{
    glm::vec3 v0 = vertices[indices.x], v1 = vertices[indices.y],
              v2 = vertices[indices.z];
    float u = rnd(seed), v = rnd(seed);
    if (u + v > 1)
    {
        u = 1 - u;
        v = 1 - v;
    }
    return v0 + u * (v1 - v0) + v * (v2 - v0);
}

/* Uniformly sample a point on triangle, with barycentric coords provided */
__device__ __forceinline__ glm::vec3 SampleTrianglePos(glm::vec3 *vertices,
                                                       glm::ivec3 indices,
                                                       unsigned int &seed,
                                                       glm::vec3 &coord)
{
    glm::vec3 v0 = vertices[indices.x], v1 = vertices[indices.y],
              v2 = vertices[indices.z];
    float u = rnd(seed), v = rnd(seed);
    if (u + v > 1)
    {
        u = 1 - u;
        v = 1 - v;
    }
    coord.x = 1 - u - v;
    coord.y = u;
    coord.z = v;
    return v0 + u * (v1 - v0) + v * (v2 - v0);
}

/* Sample only position*/
__host__ __device__ __forceinline__ void SampleAreaLightPos(
    uint32_t lightNum, DeviceAreaLight *areaLights, LightSample &ls,
    uint32_t &seed)
{
    uint32_t idx = static_cast<uint32_t>(lightNum * rnd(seed));
    auto &lt = areaLights[idx];
    float pdf = 1.0f / lightNum;
    uint32_t triIdx = static_cast<uint32_t>(rnd(seed) * lt.triangleNum);
    pdf = pdf / lt.triangleNum;
    glm::ivec3 &tri = lt.indices[triIdx];
    glm::vec3 coord;
    ls.pos = SampleTrianglePos(lt.vertices, tri, seed, coord);
    pdf = pdf / GetTriangleArea(lt.vertices, tri);
    /* Get normal */
    glm::vec3 N = BarycentricByIndices(lt.normals, tri, coord);
    ls.N = glm::normalize(N);
    ls.pdf = pdf;
    ls.areaLightID = idx;
    return;
}

} // namespace EasyRender::Device