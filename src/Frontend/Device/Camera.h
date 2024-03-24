#pragma once
#include "cuda_runtime.h"
#include "glm/glm.hpp"

using glm::ivec2, glm::vec3;

namespace Wayland
{

vec3 __host__ __device__ __forceinline__
PinholeGenerateRay(ivec2 index, ivec2 size, vec3 &lookAt, vec3 &up, vec3 &right)
{
	float u = 2.f * (index.x + 0.5f) / size.x - 1.f;
	float v = 2.f * (index.y + 0.5f) / size.y - 1.f;

	vec3 rayDir = lookAt + right * (2.0f * u - 1.0f) + up * (2.0f * v - 1.0f);
	return glm::normalize(rayDir);
}

} // namespace Wayland
