#pragma once
#include "SimpleLaunchParams.h"
#include "Core/ProgramManager.h"
#include "Core/Renderer.h"

namespace EasyRender
{

class SimpleProgramManager : public ProgramManager
{
public:
    SimpleProgramManager(Renderer *r_) : renderer(r_)
    {
    }

    void Setup()
    {
        param.frameID = 0;
        param.fbSize.x = renderer->window.size.w;
        param.fbSize.y = renderer->window.size.h;
        param.colorBuffer = (uint32_t *)renderer->device.deviceFrameBuffer;
    }

    void Update()
    {
        param.frameID += 1;
        param.fbSize.x = renderer->window.size.w;
        param.fbSize.y = renderer->window.size.h;
        param.colorBuffer = (uint32_t *)renderer->device.deviceFrameBuffer;
    }

    void *GetParamPtr() { return &param; }
    size_t GetParamSize() { return sizeof(param); };

    Optix::ShaderBindingTable GenerateSBT(const Optix::ProgramGroupArray &pg)
    {
        using namespace Optix;
        SBTData<void> raygenData{};
        SBTData<int> missData{};
        vector<SBTData<int>> hitData(1);
        std::size_t index = 2;
        return ShaderBindingTable{ raygenData,           0,     missData, 1,
                                   std::span(hitData),  &index, pg
        };
    }

private:
    Renderer *renderer;
    SimpleLaunchParams param;
};

} // namespace EasyRender
