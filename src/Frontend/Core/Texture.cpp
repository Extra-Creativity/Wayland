#include "Texture.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace EasyRender
{

Texture::Texture(minipbrt::ImageMapTexture *miniTex) 
{
    TexLoadImage(miniTex->filename);
}

void Texture::TexLoadImage(std::string_view src)
{
    data = stbi_load(src.data(), &size.x, &size.y, &channels, 4);
}

Texture::~Texture()
{
    if (data)
        stbi_image_free(data);
}

} // namespace EasyRender