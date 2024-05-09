#include <sstream>
#include <algorithm>
#include "Material.h"
#include "SceneManager.h"
#include "Utils/Common.h"
#include "Utils/PrintUtils.h"
#include "glm/gtc/type_ptr.hpp"

using namespace std;

namespace EasyRender
{

Diffuse::Diffuse(minipbrt::MatteMaterial *miniMatte)
{
    Kd = glm::make_vec3(miniMatte->Kd.value);
    textureId = miniMatte->Kd.texture;
}

string Diffuse::ToString() const
{
    ostringstream oss;
    oss << "  Type: Diffuse\n";
    oss << "  Kd:   " << PrintUtils::ToString(Kd) << "\n";
    return oss.str();
}

bool Diffuse::HasTexture()
{
    return textureId < INVALID_INDEX;
}

Disney::Disney(minipbrt::DisneyMaterial *miniDisney)
{
    color = glm::make_vec3(miniDisney->color.value);
    textureId = miniDisney->color.texture;
    metallic = miniDisney->metallic.value;
    roughness = miniDisney->roughness.value;
    sheenTint = miniDisney->sheentint.value;
    specular = 0.f;
    specularTint = miniDisney->speculartint.value;
    eta = miniDisney->eta.value;
    trans = miniDisney->spectrans.value;
    subsurface = 0.0f;
    anisotropic = miniDisney->anisotropic.value;
    sheen = miniDisney->sheen.value;
    clearcoat = miniDisney->clearcoat.value;
    clearcoatgloss = miniDisney->clearcoatgloss.value;
}

Disney::Disney(minipbrt::MatteMaterial *miniMatte)
{
    color = glm::make_vec3(miniMatte->Kd.value);
    textureId = miniMatte->Kd.texture;
    metallic = 0.0f;
    roughness = 0.5f;
    sheenTint = 0.5f;
    specular = 0.f;
    specularTint = 0.0f;
    eta = 1.5f;
    trans = 0.0f;
    subsurface = 0.0f;
    anisotropic = 0.0f;
    sheen = 0.0f;
    clearcoat = 0.0f;
    clearcoatgloss = 1.0f;
}

Disney::Disney(minipbrt::PlasticMaterial *miniPlastic)
{
    auto kd = glm::make_vec3(miniPlastic->Kd.value);
    auto ks = glm::make_vec3(miniPlastic->Ks.value);
    color = kd + ks;
    textureId = miniPlastic->Kd.texture;
    metallic = std::clamp(glm::length(ks), 0.f, 1.f);
    roughness = miniPlastic->roughness.value;
    sheenTint = 0.5f;
    specular = 0.f;
    specularTint = 0.0f;
    eta = 1.5f;
    trans = 0.0f;
    subsurface = 0.0f;
    anisotropic = 0.0f;
    sheen = 0.0f;
    clearcoat = 0.0f;
    clearcoatgloss = 1.0f;
}

Disney::Disney(minipbrt::GlassMaterial *miniGlass)
{
    glm::vec3 Kr = glm::make_vec3(miniGlass->Kr.value);
    glm::vec3 Kt = glm::make_vec3(miniGlass->Kt.value);
    color = (Kr + Kt) / 2.f;
    textureId = miniGlass->Kt.texture;
    metallic = 0.0f;
    roughness = 0.01f;
    sheenTint = 0.5f;
    specular = 0.5f;
    specularTint = 0.0f;
    eta = miniGlass->eta.value;
    trans = 1.f;
    subsurface = 0.0f;
    anisotropic = 0.0f;
    sheen = 0.0f;
    clearcoat = 0.0f;
    clearcoatgloss = 1.0f;
}

bool Disney::HasTexture()
{
    return textureId < INVALID_INDEX;
}

void Disney::ToDevice(Device::DisneyMaterial &deviceMat) {
    deviceMat.color = color;
    deviceMat.metallic = metallic;
    deviceMat.roughness = roughness;
    deviceMat.specular = specular;
    deviceMat.specularTint = specularTint;
    deviceMat.eta = eta;
    deviceMat.trans = trans;
    deviceMat.subsurface = subsurface;
    deviceMat.anisotropic = anisotropic;
    deviceMat.sheen = sheen;
    deviceMat.sheenTint = sheenTint;
    deviceMat.clearcoat = clearcoat;
    deviceMat.clearcoatgloss = clearcoatgloss;
}

} // namespace EasyRender
