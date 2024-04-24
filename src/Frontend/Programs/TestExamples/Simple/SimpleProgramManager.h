#pragma once
#include "Core/ProgramManager.h"
#include "Core/Renderer.h"
#include "SimpleLaunchParams.h"

namespace EasyRender
{

class SimpleProgramManager : public ProgramManager
{
public:
    SimpleProgramManager(Renderer *r_) : renderer(r_) {}

    void Setup()
    {
        param.frameID = 0;
        param.fbSize = renderer->window.size;
        param.colorBuffer = (uint32_t *)renderer->device.d_FrameBuffer;
    }

    void Update()
    {
        param.frameID += 1;
        param.fbSize = renderer->window.size;
        param.colorBuffer = (uint32_t *)renderer->device.d_FrameBuffer;
    }

    void End() {}

    void *GetParamPtr() { return &param; }
    size_t GetParamSize() { return sizeof(param); };

    Optix::ShaderBindingTable GenerateSBT(const Optix::ProgramGroupArray &pg)
    {
        using namespace Optix;
        SBTData<void> raygenData{};
        SBTData<int> missData{};
        std::vector<SBTData<int>> hitData(1);
        std::size_t index = 2;
        return ShaderBindingTable{ raygenData,         0,      missData, 1,
                                   hitData, &index, pg };
    }

private:
    Renderer *renderer;
    Programs::Simple::LaunchParams param;
};

} // namespace EasyRender
