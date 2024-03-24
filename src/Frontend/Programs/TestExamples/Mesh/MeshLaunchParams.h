#pragma once
#include "glm/glm.hpp"
#include "optix.h"
#include "Device/Camera.h"

namespace Wayland
{

struct MeshLaunchParams
{
    int frameID;
    glm::ivec2 fbSize;
    PinholeCamFrame camera;
    glm::u8vec4 *colorBuffer;
    OptixTraversableHandle traversable;
};

} // namespace Wayland
