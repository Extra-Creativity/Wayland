#pragma once
#include "Core/ProgramManager.h"
#include "Core/Renderer.h"
#include "PathTracingLaunchParams.h"

namespace EasyRender
{

class PathTracingProgramManager : public ProgramManager
{
public:
    PathTracingProgramManager(Renderer *r_) : renderer(r_) {}

    void Setup();
    void Update();
    void *GetParamPtr() { return &param; }
    size_t GetParamSize() { return sizeof(param); };
    Optix::ShaderBindingTable GenerateSBT(const Optix::ProgramGroupArray &pg);

private:
    Renderer *renderer;
    Programs::PathTracing::LaunchParams param;
};

} // namespace EasyRender
