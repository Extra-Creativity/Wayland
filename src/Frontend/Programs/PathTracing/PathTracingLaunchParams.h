#pragma once
#include "Device/Camera.h"
#include "Device/Light.h"
#include "glm/glm.hpp"
#include "optix.h"

namespace EasyRender::Programs::PathTracing
{

struct LaunchParams
{
    int frameID;
    glm::ivec2 fbSize;
    glm::vec4 *radianceBuffer;
    glm::u8vec4 *colorBuffer;
    
    Device::PinholeCamFrame camera;

    int areaLightCount;
    Device::DeviceAreaLight* areaLights;

    OptixTraversableHandle traversable;
};

struct MissData
{
    glm::vec3 bg_color;
};

struct HitData
{
    int twoSided;
    unsigned int meshID;
    glm::vec3 Kd;
    glm::vec3 L;
    glm::ivec3 *indices;
    glm::vec3 *normals;
};

} // namespace EasyRender
