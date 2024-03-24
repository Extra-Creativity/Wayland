#pragma once
#include "Device/Camera.h"
#include "glm/glm.hpp"
#include "optix.h"

namespace EasyRender::Programs::Color
{

struct LaunchParams
{
    int frameID;
    glm::ivec2 fbSize;
    PinholeCamFrame camera;
    glm::u8vec4 *colorBuffer;
    OptixTraversableHandle traversable;
};

struct MissData
{
    glm::vec4 bg_color;
};

struct HitData
{
    glm::vec3 Kd;
};

} // namespace EasyRender
