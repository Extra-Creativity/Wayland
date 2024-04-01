#pragma once
#include "Common.h"
#include "Random.h"
#include "Scene.h"
#include "Utils/MathConstants.h"
#include "cuda_runtime.h"
#include "glm/glm.hpp"

namespace EasyRender::Device
{

/* Pdf function corresponds to "SampleAreaLightPos" */
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

} // namespace EasyRender::Device
