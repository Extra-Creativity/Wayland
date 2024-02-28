#pragma once
#include "HostUtils/CommonHeaders.h"

#include "ContextManager.h"
#include "Module.h"

#include <vector>

class ProgramGroupArray
{
    void AddProgramGroup_(const OptixProgramGroupDesc &, std::size_t);
    void GeneralAddHitProgramGroup_(auto &&, auto &&, auto &&);

public:
    ProgramGroupArray() = default;
    ProgramGroupArray(std::size_t expectedNum)
    {
        programNames_.reserve(expectedNum);
        programGroups_.reserve(expectedNum);
    }

    ~ProgramGroupArray()
    {
        for (auto programGroup : programGroups_)
        {
            HostUtils::CheckOptixError<HostUtils::OnlyLog>(
                optixProgramGroupDestroy(programGroup));
        }
    }

    ProgramGroupArray &AddRaygenProgramGroup(const Module &module,
                                             std::string_view name);
    ProgramGroupArray &AddHitProgramGroup(
        const Module &module, const std::array<std::string_view, 3> &names);
    ProgramGroupArray &AddHitProgramGroup(
        const std::array<const Module *, 3> &modules,
        const std::array<std::string_view, 3> &names);
    ProgramGroupArray &AddMissProgramGroup(const Module &module,
                                           std::string_view name);

    ProgramGroupArray &AddRawRaygenProgramGroup(const Module &module,
                                                std::string rawName);
    ProgramGroupArray &AddRawHitProgramGroup(
        const Module &module, std::array<std::string, 3> &rawNames);
    ProgramGroupArray &AddRawHitProgramGroup(
        const std::array<const Module *, 3> &modules,
        std::array<std::string, 3> &rawNames);
    ProgramGroupArray &AddRawMissProgramGroup(const Module &module,
                                              std::string rawName);

    const auto &GetHandleArr() const noexcept { return programGroups_; }

private:
    std::vector<std::string> programNames_;
    std::vector<OptixProgramGroup> programGroups_;
};