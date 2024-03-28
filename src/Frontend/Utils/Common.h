#pragma once
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <span>

#include "glm/gtc/type_ptr.hpp"
#include "glm/glm.hpp"
#include "MathConstants.h"

namespace EasyRender
{

template<typename T, unsigned int N, typename Y>
inline auto VecToSpan(std::vector<Y> &v) -> std::span<const T, std::dynamic_extent>
{
    return { (T *)(&v[0]), v.size() * N };
}

// For point, dim4=1; for vector, dim4=0
inline glm::vec3 TransformVec3(glm::mat4 &m, glm::vec3 &v, float dim4)
{
    return m * glm::vec4(v.x, v.y, v.z, dim4);
}

// For point, dim4=1; for vector, dim4=0
inline glm::vec3 TransformVec3(glm::mat4 &m, float *v3, float dim4)
{
    return m * glm::vec4(v3[0], v3[1], v3[2], dim4);
}

inline float AngleToRadian(float a)
{
	return static_cast<float>(a * PI / 180.0);
}

} // namespace EasyRender