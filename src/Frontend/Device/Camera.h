#pragma once
#include "cuda_runtime.h"
#include "glm/glm.hpp"

using glm::ivec2, glm::vec3;

namespace EasyRender
{

struct PinholeCamFrame
{
	vec3 pos;
	vec3 lookAt;
	vec3 up;
	vec3 right;
};

vec3 __host__ __device__ __forceinline__ PinholeGenerateRay(
    ivec2 index, ivec2 size, PinholeCamFrame& cam)
{
	float u = 2.f * (index.x + 0.5f) / size.x - 1.f;
	float v = 2.f * (index.y + 0.5f) / size.y - 1.f;
	vec3 rayDir = cam.lookAt + cam.right * u + cam.up * v;
	return glm::normalize(rayDir);
}

} // namespace EasyRender
