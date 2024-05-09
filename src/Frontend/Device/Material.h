#pragma once
#include "glm/glm.hpp"

namespace EasyRender::Device
{

struct DisneyMaterial
{
    glm::vec3 color = { 0.5f, 0.5f, 0.5f };
    float metallic = 0.0f;
    float roughness = 0.5f;
    float specular = 0.0f;
    float specularTint = 0.0f;
    float eta = 1.5f;
    float trans = 0.0f;
    float subsurface = 0.0f;
    float anisotropic = 0.0f;
    float sheen = 0.0f;
    float sheenTint = 0.5f;
    float clearcoat = 0.0f;
    float clearcoatgloss = 1.0f;
};

} // namespace EasyRender::Device