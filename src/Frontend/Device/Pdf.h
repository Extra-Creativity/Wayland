#pragma once
#include "cuda_runtime.h"
#include "glm/glm.hpp"
#include "Device/Scene.h"
#include "Device/Common.h"
#include "Device/Light.h"
#include "Device/Material.h"

namespace EasyRender::Device
{

__host__ __device__ __forceinline__ void PdfAreaLightPos(
    uint32_t lightNum, DeviceAreaLight *areaLights, uint32_t primIdx,
    LightSample &ls)
{
    float pdf = 1.0f / lightNum;
    auto &lt = areaLights[ls.areaLightID];
    pdf /= lt.triangleNum;
    glm::ivec3 indices = lt.indices[primIdx];
    pdf /= GetTriangleArea(lt.vertices, indices);
    ls.pdf = pdf;
    return;
}

/* Pdf function corresponds to "SampleDisneyBSDF" */
__host__ __device__ __forceinline__ float PdfDisneyBSDF(
    const DisneyMaterial &mat, const glm::vec3 &Ns_, const glm::vec3 &Ng,
    const glm::vec3 &V, const glm::vec3 &L)
{
    using namespace MaterialMath;
    float transRatio = mat.trans;
    glm::vec3 Ns = Ns_;
    float eta = mat.eta;
    if (glm::dot(Ns, V) < 0)
    {
        eta = 1 / eta;
        Ns = -Ns;
    }
    float pdf = 0;
    float NdotV = fabsf(glm::dot(V, Ns));
    float NdotL = fabsf(glm::dot(L, Ns));
    glm::vec3 wh = -glm::normalize(V + L * eta);

    float HdotV = fabsf(glm::dot(V, wh));
    float HdotL = fabsf(glm::dot(L, wh));

    float specularAlpha = mat.roughness;
    float clearcoatAlpha = lerp(0.1f, 0.001f, mat.clearcoatgloss);

    float diffuseRatio = 0.5f * (1.f - mat.metallic); // *(1 - transRatio);
    float specularRatio = 1.f - diffuseRatio;

    glm::vec3 half;
    half = glm::normalize(L + V);

    float cosTheta = fabsf(glm::dot(half, Ns));
    float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;
    float pdfGTR1 = GTR1(cosTheta, clearcoatAlpha) * cosTheta;

    // Calculate diffuse and specular pdfs and mix ratio
    float ratio = 1.0f / (1.0f + mat.clearcoat);
    float pdfSpec =
        lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * fabsf(glm::dot(L, half)));
    float pdfDiff = fabsf(glm::dot(L, Ns)) * RECIP_PI;

    float cosThetaI = fabsf(glm::dot(wh, V));
    float sin2ThetaI = 1 - cosThetaI * cosThetaI;
    float sin2ThetaT = 1 / (eta * eta) * sin2ThetaI;

    float refractRatio = 0;
    float temp = (1 - NdotV * NdotV) / (eta * eta);
    if (temp < 1)
        refractRatio = 1 - Fresnel(NdotV, sqrtf(1 - temp), eta);

    pdf = (diffuseRatio * pdfDiff + specularRatio * pdfSpec) *
          (1 - transRatio * refractRatio); // normal reflect

    if (sin2ThetaT <= 1 && glm::dot(L, wh) * glm::dot(V, wh) < 0) // refract
    {
        // Compute change of variables _dwh\_dwi_ for microfacet transmission
        float sqrtDenom = eta * glm::dot(L, wh) + glm::dot(V, wh);
        float dwh_dwi =
            fabsf((eta * eta * glm::dot(L, wh)) / (sqrtDenom * sqrtDenom));
        float a = fmaxf(0.001f, mat.roughness);
        float Ds = GTR2(fabsf(glm::dot(wh, Ns)), a);
        float pdfTrans = Ds * fabsf(glm::dot(Ns, wh)) * dwh_dwi;
        pdf += transRatio * pdfTrans * refractRatio;
    }

    cosThetaI = fabsf(glm::dot(half, V));
    sin2ThetaI = 1 - cosThetaI * cosThetaI;
    sin2ThetaT = 1 / (eta * eta) * sin2ThetaI;

    if (sin2ThetaT > 1) // full reflect
    {
        pdf += (diffuseRatio * pdfDiff + specularRatio * pdfSpec) * transRatio *
               refractRatio;
    }

    return pdf;
}

} // namespace EasyRender::Device
