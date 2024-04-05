#pragma once
#include <memory>
#include <string>

#include "glm/glm.hpp"
#include "minipbrt.h"

namespace EasyRender
{

enum class MaterialType
{
    Diffuse
};

const std::string MaterialTypeStr[] = { "diffuse" };

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

} // namespace EasyRender