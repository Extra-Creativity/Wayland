#include "Light.h"
#include "glm/gtc/type_ptr.hpp"

namespace EasyRender
{

AreaLight::AreaLight(minipbrt::DiffuseAreaLight *miniLight)
{
    L = glm::make_vec3(miniLight->L);
}

} // namespace EasyRender