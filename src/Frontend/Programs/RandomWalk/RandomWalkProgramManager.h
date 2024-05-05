#pragma once
#include "Core/ProgramManager.h"
#include "Core/Renderer.h"
#include "RandomWalkLaunchParams.h"

namespace EasyRender
{

class RandomWalkProgramManager : public ProgramManager
{
public:
    RandomWalkProgramManager(Renderer *r_) : renderer(r_) {}

    void Setup();
    void Update();
    void End();
    void *GetParamPtr() { return &param; }
    size_t GetParamSize() { return sizeof(param); };
    Optix::ShaderBindingTable GenerateSBT(const Optix::ProgramGroupArray &pg);

private:
    Renderer *renderer;
    Programs::RandomWalk::LaunchParams param;
};

} // namespace EasyRender
