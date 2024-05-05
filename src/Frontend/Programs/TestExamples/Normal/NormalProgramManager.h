#pragma once
#include "Core/ProgramManager.h"
#include "Core/Renderer.h"
#include "NormalLaunchParams.h"

namespace EasyRender
{

class NormalProgramManager : public ProgramManager
{
public:
    NormalProgramManager(Renderer *r_) : renderer(r_) {}

    void Setup();
    void Update();
    void End();
    void *GetParamPtr() { return &param; }
    size_t GetParamSize() { return sizeof(param); };
    Optix::ShaderBindingTable GenerateSBT(const Optix::ProgramGroupArray &pg);

private:
    Renderer *renderer;
    Programs::Normal::LaunchParams param;
};

} // namespace EasyRender
