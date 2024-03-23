#pragma once
#include "glm/glm.hpp"
#include "optix.h"

namespace Wayland
{

struct DepthLaunchParams
{
    int frameID;
    
    struct
    {
        glm::vec3 pos;
        glm::vec3 lookAt;
        glm::vec3 up;
        glm::vec3 right;
    } camera;

    struct
    {
        int x;
        int y;
    } fbSize;
    
    
    float minDepth;
    float maxDepth;
    float *depthBuffer;
    glm::u8vec4 *colorBuffer;

    OptixTraversableHandle traversable;
};

} // namespace Wayland
