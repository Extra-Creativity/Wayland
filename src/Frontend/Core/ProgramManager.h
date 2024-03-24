#pragma once
#include "DeviceManager.h"

namespace EasyRender
{
class ProgramManager
{
public:
    ProgramManager() = default;

public:
    virtual void Setup() = 0;
    virtual void Update() = 0;
    virtual void *GetParamPtr() = 0;
    virtual size_t GetParamSize() = 0;
    virtual Optix::ShaderBindingTable GenerateSBT(
        const Optix::ProgramGroupArray &pg) = 0;
};

using ProgramManagerPtr = std::unique_ptr<ProgramManager>;

} // namespace EasyRender