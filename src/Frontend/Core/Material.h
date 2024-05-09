#pragma once
#include <memory>
#include <string>

#include "glm/glm.hpp"
#include "minipbrt.h"
#include "Device/Material.h"

namespace EasyRender
{

enum class MaterialType
{
    Diffuse,
    Disney
};

const std::string MaterialTypeStr[] = { "Diffuse", "Disney" };

class Material
{
public:
    Material() = default;
    ~Material() = default;
    virtual MaterialType type() = 0;
    virtual bool HasTexture() = 0;
    virtual std::string ToString() const = 0;
};

using MaterialPtr = std::unique_ptr<Material>;

class Diffuse : public Material
{
public:
    Diffuse(minipbrt::MatteMaterial *);
    ~Diffuse() = default;
    MaterialType type() { return MaterialType::Diffuse; }
    bool HasTexture();
    std::string ToString() const;

    glm::vec3 Kd;
    uint32_t textureId;
};

class Disney : public Material
{
public:
    Disney(minipbrt::DisneyMaterial *);
    Disney(minipbrt::MatteMaterial *);
    Disney(minipbrt::PlasticMaterial *);
    Disney(minipbrt::GlassMaterial *);
    ~Disney() = default;
    MaterialType type() { return MaterialType::Disney; }
    bool HasTexture();
    std::string ToString() const { return std::string("\n"); }
    void ToDevice(Device::DisneyMaterial& deviceMat);

//private:
    glm::vec3 color;
    float metallic;
    float roughness;
    float specular;
    float specularTint;
    float eta;
    float trans;
    float subsurface;
    float anisotropic;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatgloss;

    uint32_t textureId;
};

} // namespace EasyRender