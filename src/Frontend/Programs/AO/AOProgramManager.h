#pragma once
#include "Core/ProgramManager.h"
#include "Core/Renderer.h"
#include "AOLaunchParams.h"

namespace EasyRender
{

class AOProgramManager : public ProgramManager
{
public:
    AOProgramManager(Renderer *r_) : renderer(r_) {}

    void Setup();
    void Update();
    void End();
    void *GetParamPtr() { return &param; }
    size_t GetParamSize() { return sizeof(param); };
    Optix::ShaderBindingTable GenerateSBT(const Optix::ProgramGroupArray &pg);

private:
    Renderer *renderer;
    Programs::AO::LaunchParams param;
};

} // namespace EasyRender
