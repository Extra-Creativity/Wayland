#include "DeviceUtils/Camera.h"
#include "cuda_runtime.h"
#include "glm/glm.hpp"
#include "optix_types.h"

namespace Wayland::Example::Phong
{

struct LaunchParam
{
    glm::vec3 lightPos;
    glm::vec3 lightColor;
    Wayland::Example::DeviceUtils::Camera camera;
    glm::u8vec4 *colorBuffer;
    OptixTraversableHandle traversable;
};

struct HitData
{
    glm::vec3 kd;
    glm::vec3 ks;
    glm::vec3 ka;
    int sPow;
    cudaTextureObject_t diffuseTexture;
    glm::ivec3 *indices;
    glm::vec2 *texCoords;
    glm::vec3 *normals;
};

struct MissData
{
    glm::vec3 bgColor;
};

} // namespace Wayland::Example::Phong