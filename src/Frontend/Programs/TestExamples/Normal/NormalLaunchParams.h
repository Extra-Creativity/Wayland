#pragma once
#include "Device/Camera.h"
#include "glm/glm.hpp"
#include "optix.h"

namespace EasyRender::Programs::Normal
{

struct LaunchParams
{
    int frameID;
    glm::ivec2 fbSize;
    Device::PinholeCamFrame camera;
    glm::u8vec4 *colorBuffer;
    OptixTraversableHandle traversable;
};

struct MissData
{
    glm::vec3 bg_color;
};

struct HitData
{
    unsigned int meshID; 
    glm::vec3 Kd;
    glm::ivec3 *indices;
    glm::vec3 *vertices;
    glm::vec3 *normals;
    bool hasNormal;
};

} // namespace EasyRender
