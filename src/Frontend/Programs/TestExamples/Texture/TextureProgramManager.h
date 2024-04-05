#pragma once
#include "TextureLaunchParams.h"
#include "Core/ProgramManager.h"
#include "Core/Renderer.h"

namespace EasyRender
{

class TextureProgramManager : public ProgramManager
{
public:
    TextureProgramManager(Renderer *r_) : renderer(r_) {}

    void Setup();
    void Update();
    void End(){};
    void *GetParamPtr() { return &param; }
    size_t GetParamSize() { return sizeof(param); };
    Optix::ShaderBindingTable GenerateSBT(const Optix::ProgramGroupArray &pg);

private:
    Renderer *renderer;
    Programs::Texture::LaunchParams param;
};

} // namespace EasyRender
