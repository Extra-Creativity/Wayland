#pragma once
#include "Core/ProgramManager.h"
#include "Core/Renderer.h"
#include "ColorLaunchParams.h"

namespace EasyRender
{

class ColorProgramManager : public ProgramManager
{
public:
    ColorProgramManager(Renderer *r_) : renderer(r_) {}

    void Setup();
    void Update();
    void *GetParamPtr() { return &param; }
    size_t GetParamSize() { return sizeof(param); };
    Optix::ShaderBindingTable GenerateSBT(const Optix::ProgramGroupArray &pg);

private:
    Renderer *renderer;
    Programs::Color::LaunchParams param;
};

} // namespace EasyRender
