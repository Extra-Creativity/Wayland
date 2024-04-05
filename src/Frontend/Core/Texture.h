#pragma once
#include "glm/glm.hpp"
#include "minipbrt.h"
#include <string_view>
#include <memory>

namespace EasyRender
{

class Texture
{
public:
    Texture() : size{}, channels(0), data(nullptr) {}
    Texture(minipbrt::ImageMapTexture *miniTex);
    ~Texture();

private:
    void TexLoadImage(std::string_view src);

public:
    glm::ivec2 size;
    int32_t channels;
    uint8_t *data;
};


using TexturePtr = std::unique_ptr<Texture>;

} // namespace EasyRender