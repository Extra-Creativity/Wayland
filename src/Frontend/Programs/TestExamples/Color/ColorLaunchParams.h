#pragma once
#include "Device/Camera.h"
#include "glm/glm.hpp"
#include "optix.h"

namespace EasyRender
{

struct ColorLaunchParams
{
    int frameID;
    glm::ivec2 fbSize;
    PinholeCamFrame camera;
    glm::u8vec4 *colorBuffer;
    OptixTraversableHandle traversable;
};

} // namespace EasyRender
