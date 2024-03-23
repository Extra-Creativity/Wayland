#include "Core/Camera.h"
#include "Utils/PrintUtils.h"
#include <glm/gtc/type_ptr.hpp>
#include <sstream>


using namespace std;
using glm::mat4, glm::vec3, glm::vec4, glm::normalize;

namespace Wayland
{

PinholeCamera::PinholeCamera(float fov_, minipbrt::Transform cameraToWorld)
    : fov(fov_)
{
    mat4 t = glm::transpose(glm::make_mat4(&cameraToWorld.start[0][0]));
    /* Use glm vec4 to vec3 conversion */
    position = t*vec4(0, 0, 0, 1);
    up = t * normalize(vec4(0, 1, 0, 0));
    lookAt = t*vec4(0, 0, 1, 0);
    right = normalize(t*vec4(1, 0, 0, 0));

    lookAt = normalize(lookAt - position);
    up = glm::tan(fov/2) * up;
    right = glm::tan(fov/2) * right;
}

PinholeCamera::PinholeCamera(float fov_, vec3 pos_, vec3 lookAt_, vec3 up_)
    : fov(fov_), position(pos_), lookAt(lookAt_), up(up_)
{
}

string PinholeCamera::ToString() const
{
    ostringstream oss;
    oss << "  Type:     Pinhole\n";
    oss << "  Position: " << PrintUtils::ToString(position) << "\n";
    oss << "  Lookat:   " << PrintUtils::ToString(lookAt) << "\n";
    oss << "  Up:       " << PrintUtils::ToString(up) << "\n";
    oss << "  Fov:      " << fov << "\n";
    return oss.str();
}
} // namespace Wayland
