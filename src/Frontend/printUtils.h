#pragma once
#include <glm/mat4x4.hpp>
#include <iostream>
#include <sstream>
#include <string>

#include "minipbrt.h"
#include "PBRTv3_SceneCommon.h"

using namespace std;

namespace Wayland::PrintUtils
{

string ToString(glm::vec3 v);
string ToString(minipbrt::Scene *scene);
string ToString(minipbrt::Shape *shape);

} // namespace Wayland