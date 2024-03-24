#pragma once
#include "Device/Camera.h"
#include "glm/glm.hpp"
#include "optix.h"

namespace EasyRender::Programs::PathTracing
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
    glm::vec3 bg_color;
};

struct HitData
{
    glm::vec3 Kd;
    glm::vec3 L;
    glm::ivec3 *indices;
    glm::vec3 *normals;
};

} // namespace EasyRender
