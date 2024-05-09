#pragma once
#include "cuda_runtime.h"
#include "Utils/MathConstants.h"
#include "Device/Material.h"
#include "Device/Common.h"

namespace EasyRender::Device
{

namespace MaterialMath
{

__host__ __device__ __forceinline__ float SchlickFresnel(float u)
{
    float m = clamp(1.0f - u, 0.0f, 1.0f);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

__host__ __device__ __forceinline__ float Fresnel(float cos_theta_i,
                                                 float cos_theta_t,
                                         float eta)
{
    const float rs =
        (cos_theta_i - cos_theta_t * eta) / (cos_theta_i + eta * cos_theta_t);
    const float rp =
        (cos_theta_i * eta - cos_theta_t) / (cos_theta_i * eta + cos_theta_t);
    return 0.5f * (rs * rs + rp * rp);
}

__host__ __device__ __forceinline__ float GTR1(float NDotH, float a)
{
    if (a >= 1.0f)
        return (1.0f / PI);
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
    return (a2 - 1.0f) / (PI * logf(a2) * t);
}

__host__ __device__ __forceinline__ float GTR2(float NDotH, float a)
{
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
    return a2 / (PI * t * t);
}

__host__ __host__ __device__ __forceinline__ float SmithG_GGX(float NDotv,
                                                             float alphaG)
{
    float a = alphaG * alphaG;
    float b = NDotv * NDotv;
    return 1.0f / (NDotv + sqrtf(a + b - a * b));
}

__host__ __device__ __forceinline__ float Lambda(const glm::vec3 &w,
                                                const glm::vec3 &n)
{
    // BeckmannDistribution
    const float alphax = 0.5f;
    // const float alphay = 0.5f;
    float wDotn = glm::dot(w, n);
    float absTanTheta = fabsf(glm::length(glm::cross(w, n)) / wDotn);
    if (isinf(absTanTheta))
        return 0.;
    // Compute _alpha_ for direction _w_
    float alpha = alphax;
    float a = 1 / (alpha * absTanTheta);
    if (a >= 1.6f)
        return 0;
    return (1 - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
}

__host__ __device__ __forceinline__ glm::vec3 EvalDisneyTransmit(
    const DisneyMaterial &mat, const glm::vec3 &Ns, const glm::vec3 &Ng,
    const glm::vec3 &V_vec,
    const glm::vec3 &L_vec,
    glm::vec3 texColor = glm::vec3{ 1.f, 1.f, 1.f }
)
{
    glm::vec3 N = Ns;
    glm::vec3 V = V_vec;
    glm::vec3 L = L_vec;
    glm::vec3 combinedColor = texColor * mat.color;
    float NDotL = glm::dot(N, L);
    float NDotV = glm::dot(N, V);
    float mateta = mat.eta;
    float eta = 1 / mateta;

    if (NDotL > 0)
    {
        eta = 1 / eta;
        N = -N;
    }

    if (NDotL == 0 || NDotV == 0)
        return glm::vec3{ 0.f, 0.f, 0.f };
    float refract;
    if ((1 - NDotV * NDotV) * eta * eta >= 1) // ȫ
        refract = 1;
    else
        refract = 0;
    glm::vec3 Cdlin = combinedColor;
    float Cdlum =
        0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z; // luminance approx.
    auto Ctint =
        Cdlum > 0.0f
            ? Cdlin / Cdlum
            : glm::vec3(1.f, 1.f, 1.f); // normalize lum. to isolate hue+sat
    auto Cspec0 = lerp(
        mat.specular * 0.08f *
            lerp(glm::vec3(1.f, 1.f, 1.f), Ctint, mat.specularTint),
        Cdlin, mat.metallic);

    // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
    auto wh = glm::normalize(L + V * eta);
    if (glm::dot(wh, N) < 0)
        wh = -wh;

    // Same side?
    if (glm::dot(L, wh) * glm::dot(V, wh) > 0)
        return glm::vec3{ 0.f, 0.f, 0.f };

    float sqrtDenom = glm::dot(L, wh) + eta * glm::dot(V, wh);
    float factor = 1 / eta;
    auto T = mat.trans * combinedColor;
    float roughg = sqrtf(mat.roughness * 0.5f + 0.5f);
    float Gs = 1 / (1 + Lambda(V, N) + Lambda(L, N));
    float a = fmaxf(0.001f, mat.roughness);
    float Ds = GTR2(glm::dot(N, wh), a);
    float FH = SchlickFresnel(glm::dot(V, wh));
    glm::vec3 Fs = lerp(Cspec0, glm::vec3(1.f, 1.f, 1.f), FH);
    float F = Fresnel(fabsf(glm::dot(V, wh)), fabsf(glm::dot(L, wh)), eta);
    glm::vec3 out = (1 - refract) * (1.f - F) * T *
                    fabsf(Ds * Gs * eta * eta * fabsf(glm::dot(L, wh)) *
                          fabsf(glm::dot(V, wh)) * factor * factor /
                          (NDotL * NDotL * sqrtDenom * sqrtDenom));
    return out;
}

} // namespace MaterialMath

__host__ __device__ __forceinline__ glm::vec3 EvalDisneyBSDF(
    const DisneyMaterial &mat, const glm::vec3 &Ns_,
                         const glm::vec3 &Ng, const glm::vec3 &V,
                         const glm::vec3 &L,
                         glm::vec3 texColor = glm::vec3{ 1.f, 1.f, 1.f })
{
    using namespace MaterialMath;

    glm::vec3 combinedColor = texColor * mat.color;
    glm::vec3 Ns = Ns_;
    float mateta = mat.eta;
    float NDotL = glm::dot(Ns, L);
    float gNDotL = glm::dot(Ng, L);
    float NDotV = glm::dot(Ns, V);
    float gNDotV = glm::dot(Ng, V);
    float eta = 1 / mateta;
    if (mat.trans < 0.1 && (gNDotL * gNDotV <= 0.0f || NDotL * NDotV <= 0))
        return glm::vec3{ 0.f, 0.f, 0.f };
    if (mat.trans > 0.9 && (gNDotL * gNDotV * NDotL * NDotV <= 0))
        return glm::vec3{ 0.f, 0.f, 0.f };
    if (NDotL * NDotV <= 0)
        return EvalDisneyTransmit(mat, Ns, Ng, V, L, texColor);
    if (mat.metallic + combinedColor.x + combinedColor.y + combinedColor.z <= 0)
        return glm::vec3{ 0.f, 0.f, 0.f };

    if (NDotL < 0.0f && NDotV < 0.0f)
    {
        Ns = -Ns;
        eta = 1 / eta;
        NDotL *= -1;
        NDotV *= -1;
    }
    glm::vec3 H = glm::normalize(L + V);
    float NDotH = glm::dot(Ns, H);
    float LDotH = glm::dot(L, H);
    float VDotH = glm::dot(V, H);

    float refract;
    if ((1 - NDotV * NDotV) * eta * eta >= 1)
        refract = 1;
    else
        refract = 0;

    glm::vec3 Cdlin = combinedColor;
    float Cdlum =
        0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z; // luminance approx.

    glm::vec3 Ctint =
        Cdlum > 0.0f
            ? Cdlin / Cdlum
            : glm::vec3{ 1.f, 1.f, 1.f }; // normalize lum. to isolate hue+sat
    glm::vec3 Cspec0 = lerp(
        mat.specular * 0.08f *
            lerp(glm::vec3{ 1.f, 1.f, 1.f }, Ctint, mat.specularTint),
        Cdlin, mat.metallic);
    glm::vec3 Csheen =
        lerp(glm::vec3{ 1.f, 1.f, 1.f }, Ctint, mat.sheenTint);

    // Diffuse fresnel - go from 1 at Ng incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
    float Fd90 = 0.5f + 2.0f * LDotH * LDotH * mat.roughness;
    float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotrokPic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LDotH * LDotH * mat.roughness;
    float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.0f / (NDotL + NDotV) - 0.5f) + 0.5f);

    // specular
    // float aspect = sqrt(1-mat.anisotrokPic*.9);
    // float ax = Max(.001f, sqr(mat.roughness)/aspect);
    // float ay = Max(.001f, sqr(mat.roughness)*aspect);
    // float Ds = GTR2_aniso(NDotH, Dot(H, X), Dot(H, Y), ax, ay);

    float a = fmaxf(0.001f, mat.roughness);
    float Ds = GTR2(NDotH, a);
    float FH = SchlickFresnel(LDotH);
    glm::vec3 Fs = lerp(Cspec0, glm::vec3{ 1.f, 1.f, 1.f }, FH);
    float roughg = sqrtf(mat.roughness * 0.5f + 0.5f);
    float Gs = SmithG_GGX(NDotL, roughg) * SmithG_GGX(NDotV, roughg);

    // sheen
    glm::vec3 Fsheen = FH * mat.sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, mat.clearcoatgloss));
    float Fr = lerp(0.04f, 1.0f, FH);
    float Gr = SmithG_GGX(NDotL, 0.25f) * SmithG_GGX(NDotV, 0.25f);

    float trans = mat.trans;

    float cosThetaI = fabsf(glm::dot(Ns, V));
    float sin2ThetaI = 1 - cosThetaI * cosThetaI;
    float sin2ThetaT = eta * eta * sin2ThetaI;
    float cosThetaT = 1;
    if (sin2ThetaT <= 1)
    {
        cosThetaT = sqrt(1 - sin2ThetaT);
    }
    float F = Fresnel(cosThetaI, cosThetaT, eta);

    glm::vec3 out =
        (((1.0f / PI) * lerp(Fd, ss, mat.subsurface) * Cdlin + Fsheen) *
         (1.0f - mat.metallic)) *
        (1 - trans * (1 - F) * (1 - refract));
    if (trans > 0)
        out = out + Gs * Ds * (1 - trans * (1 - refract) * (1 - F));
    else
        out = out + Gs * Ds * Fs;
    return out;
}

} // namespace EasyRender::Device