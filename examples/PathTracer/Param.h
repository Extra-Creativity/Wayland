#include "DeviceUtils/Camera.h"
#include "cuda_runtime.h"
#include "glm/glm.hpp"
#include "optix_types.h"

// #include <numbers>

namespace Wayland::Example::PathTracing
{

struct LaunchParam
{
    unsigned int sampleNum;
    unsigned int maxDepth;
    Wayland::Example::DeviceUtils::Camera camera;
    glm::vec4 *colorBuffer;
    OptixTraversableHandle traversable;
};

struct HitData
{
    glm::vec3 color;
    glm::vec3 emission;
    // cornell box doesn't have a texture; for general case, if(hasTexture) may
    // be needed.
    // cudaTextureObject_t diffuseTexture;
    glm::ivec3 *indices;
    glm::vec3 *normals;
};

} // namespace Wayland::Example::PathTracing