#pragma once
#include "Device/Camera.h"
#include "Device/Light.h"
#include "glm/glm.hpp"
#include "optix.h"

namespace EasyRender::Programs::PathTracing
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
    glm::vec3 throughput;
    glm::vec3 rayPos;
    glm::vec3 rayDir;
    /* Normal of last hit surface*/
    glm::vec3 lastNormal;
    /* cosine / sample pdf*/
    float lastTraceTerm;
    unsigned int depth;
    unsigned int seed;
    bool done;
};

struct LaunchParams
{
    int frameID;
    glm::ivec2 fbSize;
    glm::vec4 *radianceBuffer;
    glm::u8vec4 *colorBuffer;

    Device::PinholeCamFrame camera;

    int areaLightCount;
    Device::DeviceAreaLight *areaLights;

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

} // namespace EasyRender::Programs::PathTracing
