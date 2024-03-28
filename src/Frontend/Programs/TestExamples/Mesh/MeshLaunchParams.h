#pragma once
#include "glm/glm.hpp"
#include "optix.h"
#include "Device/Camera.h"

namespace EasyRender::Programs::Mesh
{

struct LaunchParams
{
    int frameID;
    glm::ivec2 fbSize;
    Device::PinholeCamFrame camera;
    glm::u8vec4 *colorBuffer;
    OptixTraversableHandle traversable;
};

} // namespace EasyRender
