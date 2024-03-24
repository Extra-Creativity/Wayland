#pragma once
#include "glm/glm.hpp"
#include "optix.h"

namespace EasyRender::Programs::Simple
{

struct LaunchParams
{
    int frameID;
    glm::ivec2 fbSize;
    uint32_t *colorBuffer;
};

} // namespace EasyRender
