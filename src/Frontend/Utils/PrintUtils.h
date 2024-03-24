#pragma once
#include <glm/mat4x4.hpp>
#include <iostream>
#include <sstream>
#include <string>

#include "minipbrt.h"
#include "Utils/PBRTv3_SceneCommon.h"

namespace EasyRender::PrintUtils
{

std::string ToString(glm::vec3 v);
std::string ToString(minipbrt::Scene *scene);
std::string ToString(minipbrt::Material *material);
std::string ToString(minipbrt::Shape *shape);

} // namespace EasyRender