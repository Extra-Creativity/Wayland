#pragma once
#include <glm/mat4x4.hpp>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

namespace Wayland::printUtils
{

string toString(glm::vec3 v)
{
    ostringstream oss;
    oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return oss.str();
}

} // namespace Wayland