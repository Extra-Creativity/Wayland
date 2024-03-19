#include "Camera.h"
#include "printUtils.h"
#include <glm/gtc/type_ptr.hpp>
#include <sstream>


using namespace std;
using glm::mat4, glm::vec3, glm::vec4;

namespace Wayland
{

PinholeCamera::PinholeCamera(float fov_, minipbrt::Transform cameraToWorld)
    : fov(fov_)
{
    mat4 t = glm::make_mat4(&cameraToWorld.start[0][0]);
    /* Use glm vec4 to vec3 conversion */
    position = t * vec4(0, 0, 0, 1);
    up = t * vec4(0, 1, 0, 0);
    lookAt = t * vec4(0, 0, 1, 0);
}

PinholeCamera::PinholeCamera(float fov_, vec3 pos_, vec3 lookAt_, vec3 up_)
    : fov(fov_), position(pos_), lookAt(lookAt_), up(up_)
{
}

string PinholeCamera::toString() const
{
    ostringstream oss;
    oss << "  Type:     Pinhole\n";
    oss << "  Position: " << printUtils::toString(position) << "\n";
    oss << "  Lookat:   " << printUtils::toString(lookAt) << "\n";
    oss << "  Up:       " << printUtils::toString(up) << "\n";
    oss << "  Fov:      " << fov << "\n";
    return oss.str();
}
} // namespace Wayland
