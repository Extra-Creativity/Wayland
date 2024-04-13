#pragma once
#include "glm/glm.hpp"

namespace Wayland::Example::DeviceUtils
{

struct Camera
{
    float horizontalRatio;
    float verticalRatio;
    glm::vec3 position;
    glm::vec3 gaze;
    glm::vec3 up;
};

} // namespace Wayland::Example::DeviceUtils