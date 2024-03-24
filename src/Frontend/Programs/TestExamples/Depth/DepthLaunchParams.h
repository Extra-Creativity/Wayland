#pragma once
#include "glm/glm.hpp"
#include "optix.h"
#include "Device/Camera.h"

namespace Wayland
{

struct DepthLaunchParams
{
    int frameID;
    PinholeCamFrame camera;
    glm::ivec2 fbSize;    
    float minDepth;
    float maxDepth;
    float *depthBuffer;
    glm::u8vec4 *colorBuffer;

    OptixTraversableHandle traversable;
};

} // namespace Wayland
