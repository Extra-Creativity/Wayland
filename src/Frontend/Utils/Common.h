#pragma once
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "Core/shape.h"
#include "glm/gtc/type_ptr.hpp"
#include "glm/mat4x4.hpp"

using namespace std;

namespace Wayland
{

template<typename T, unsigned int N, typename Y>
inline auto VecToSpan(vector<Y> &v) -> span<const T, dynamic_extent>
{
    return { (T *)(&v[0]), v.size() * N };
}

} // namespace Wayland