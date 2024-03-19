#pragma once
#include <glm/mat4x4.hpp>
#include <iostream>
#include <sstream>
#include <string>

#include "minipbrt.h"
#include "PBRTv3_SceneCommon.h"

using namespace std;

namespace Wayland::printUtils
{

string toString(glm::vec3 v);
string toString(minipbrt::Scene *scene);
string toString(minipbrt::Shape *shape);

} // namespace Wayland