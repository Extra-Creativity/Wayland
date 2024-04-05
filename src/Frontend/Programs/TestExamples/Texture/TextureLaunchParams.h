#pragma once
#include "Device/Camera.h"
#include "glm/glm.hpp"
#include "optix.h"

namespace EasyRender::Programs::Texture
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
    glm::vec4 bg_color;
};

struct HitData
{
    bool hasTexture;
    cudaTextureObject_t texture;
    glm::vec2 *texcoord;
    glm::ivec3 *index;
    glm::vec3 Kd;
};

} // namespace EasyRender
