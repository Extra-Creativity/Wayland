#pragma once
#include "cuda_runtime.h"
#include "glm/glm.hpp"
#include "Device/Common.h"
#include "Device/Light.h"
#include "Device/Material.h"
#include "Device/Random.h"

namespace EasyRender::Device
{

/* Uniformly samples a disk */
__device__ __forceinline__ glm::vec3 SampleUniformDisk(
    glm::vec3 pos, glm::vec3 normal, float &pdf, unsigned int &seed)
{
    glm::vec3 x, y;
	GetOrthoNormalBasis(normal, x, y);

	float r = sqrtf(rnd(seed)), theta = rnd(seed) * TWO_PI;
    pdf = 1.0f / PI;
	return pos + r * (cosf(theta) * x + sinf(theta) * y);
}

__device__ __forceinline__ glm::vec2 SampleUniformDisk(unsigned int &seed)
{
    float r = sqrtf(rnd(seed)), theta = rnd(seed) * TWO_PI;
	return glm::vec2{r * cosf(theta), r * sinf(theta)};
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

/* Samples a cosine weighted hemisphere */
__device__ __forceinline__ glm::vec3 SampleCosineHemisphere(glm::vec3 normal,
                                                            unsigned int &seed)
{
    /* Use Malley' s Method: SampleUniformDisk and project to hemisphere. */
    glm::vec3 x, y;
    GetOrthoNormalBasis(normal, x, y);
    glm::vec2 dPos = SampleUniformDisk(seed);
    float z = sqrtf(1 - dPos.x * dPos.x - dPos.y * dPos.y);
    return x * dPos.x + y * dPos.y + normal * z;
}

/* Samples a cosine weighted hemisphere */
__device__ __forceinline__ glm::vec3 SampleCosineHemisphere(glm::vec3 normal,
                                                       float &pdf,
                                                       unsigned int &seed)
{
	/* Use Malley' s Method: SampleUniformDisk and project to hemisphere. */
    glm::vec3 x, y;
    GetOrthoNormalBasis(normal, x, y);
    glm::vec2 dPos = SampleUniformDisk(seed);
    float z = sqrtf(1 - dPos.x * dPos.x - dPos.y * dPos.y);
    pdf = z / PI;
    return x * dPos.x + y * dPos.y + normal * z;
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
    if (lt.hasNormal)
	{
		glm::vec3 N = BarycentricByIndices(lt.normals, tri, coord);
		ls.N = glm::normalize(N);
    }
    else
    {
        ls.N = glm::normalize(glm::cross(lt.vertices[tri.y] - lt.vertices[tri.x],
										 lt.vertices[tri.z] - lt.vertices[tri.x]));
    }
    ls.pdf = pdf;
    ls.areaLightID = idx;
    return;
}

__host__ __device__ __forceinline__ glm::vec3 SampleDisneyBSDF(
    const DisneyMaterial &mat, const glm::vec3 &Ns_, const glm::vec3 &Ng,
                                                  const glm::vec3 &V,
                                                  uint32_t &seed)
{
    using namespace MaterialMath;
    float r1 = rnd(seed);
    float r2 = rnd(seed);
    float r3 = rnd(seed);
    float r4 = rnd(seed);
    glm::vec3 Ns = Ns_;
    float mateta = mat.eta;
    float eta = 1 / mateta;

    if (glm::dot(Ns, V) < 0)
    {
        eta = 1 / eta;
        Ns = -Ns;
    }

    float NdotV = fabsf(glm::dot(Ns, V));
    float transRatio = mat.trans;
    float transprob = rnd(seed);
    float refractprob = rnd(seed);
    float probability = rnd(seed);
    float diffuseRatio = 0.5f * (1.0f - mat.metallic); // *(1 - transRatio);
    if (transprob < transRatio)                        // sample transmit
    {
        float refractRatio = 0;
        float temp = (1 - NdotV * NdotV) * (eta * eta);
        if (temp < 1)
            refractRatio = 1 - Fresnel(NdotV, sqrtf(1 - temp), eta);
        if (refractprob < refractRatio)
        {
            glm::vec3 Ox, Oy;
            GetOrthoNormalBasis(Ns, Ox, Oy);
            float a = mat.roughness;
            float phi = r3 * TWO_PI;
            float cosTheta = sqrtf((1.0f - r4) / (1.0f + (a * a - 1.0f) * r4));
            float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);
            glm::vec3 half =
                glm::vec3{ sinTheta * cosPhi, sinTheta * sinPhi, cosTheta };
            half = half.x * Ox + half.y *Oy + half.z * Ns;

            if (glm::dot(V, Ns) == 0)
                return -V;

            float cosThetaI = glm::dot(half, V);
            float sin2ThetaI = 1 - cosThetaI * cosThetaI;
            float sin2ThetaT = eta * eta * sin2ThetaI;

            if (sin2ThetaT <= 1)
            {
                float cosThetaT = sqrtf(1 - sin2ThetaT);
                glm::vec3 L =
                    glm::normalize(eta * -V + (eta * cosThetaI - cosThetaT) * half);

                if (cosThetaI < 0) // V and half on different sided, require VdotH*LdotH<0
                {
                    float LdotH = glm::dot(L, half);
                    L = L - 2 * LdotH * half;
                }
                return L;
            }
            else
            {
                return half * 2.f * glm::dot(half, V) - V;
            }
        }
    }

    glm::vec3 Ox, Oy;
    GetOrthoNormalBasis(Ns, Ox, Oy);

    glm::vec3 dir;
    if (probability < diffuseRatio) // sample diffuse
    {
        dir = SampleCosineHemisphere(Ns, seed);\
    }
    else
    {
        float a = mat.roughness;
        float phi = r1 * TWO_PI;
        float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
        float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
        float sinPhi = sinf(phi);
        float cosPhi = cosf(phi);

        glm::vec3 half =
            glm::vec3{ sinTheta * cosPhi, sinTheta * sinPhi, cosTheta };
        half = half.z * Ns + half.x * Ox + half.y * Oy;

        dir = 2.0f * glm::dot(V, half) * half - V; // reflection vector
    }
    return dir;
}


} // namespace EasyRender::Device