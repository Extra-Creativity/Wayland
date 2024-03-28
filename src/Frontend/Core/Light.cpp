#include "glm/gtc/type_ptr.hpp"
#include "Light.h"
#include "SceneManager.h"

namespace EasyRender
{

AreaLight::AreaLight(minipbrt::DiffuseAreaLight *miniLight)
    : mesh(INVALID_INDEX)
{
    L = glm::make_vec3(miniLight->L);
    twoSided = miniLight->twosided;
}

} // namespace EasyRender