#include "glm/glm.hpp"
#include "optix.h"

struct LaunchParams
{
    glm::u8vec4 *colorBuffer;

    struct
    {
        glm::vec3 position;
        glm::vec3 gaze;
        glm::vec3 up;
    } camera;

    OptixTraversableHandle traversable;
};

struct MissData
{
    glm::vec4 bg_color;
};

struct HitData
{
    glm::vec3 emission_color;
    glm::vec3 diffuse_color;
};