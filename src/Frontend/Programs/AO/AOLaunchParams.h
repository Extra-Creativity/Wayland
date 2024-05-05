#pragma once
#include "Device/Camera.h"
#include "Device/Light.h"
#include "glm/glm.hpp"
#include "optix.h"

namespace EasyRender::Programs::AO
{

/* Ray types */
enum
{
    RADIANCE_TYPE = 0,
    SHADOW_TYPE,
    RAY_TYPE_COUNT
};

/* OptixTrace Payload */
struct Payload
{
    glm::vec3 radiance;
    glm::vec3 hitPos;
    glm::vec3 rayDir;
    glm::vec3 N;
    uint32_t seed;
};

struct LaunchParams
{
    uint32_t frameID;
    glm::ivec2 fbSize;
    glm::vec4 *radianceBuffer;
    glm::u8vec4 *colorBuffer;
    Device::PinholeCamFrame camera;
    uint32_t areaLightCount;
    Device::DeviceAreaLight *areaLights;

    OptixTraversableHandle traversable;
};

struct MissData
{
    glm::vec3 bg_color;
};

struct HitData
{
    bool hasTexture;
    bool hasNormal;
    cudaTextureObject_t texture;
    uint32_t meshID;
    uint32_t areaLightID;
    uint32_t materialID;
    glm::ivec3 *indices;
    glm::vec3 *vertices;
    glm::vec3 *normals;
    glm::vec2 *texcoords;
};

} // namespace EasyRender::Programs::PathTracing
