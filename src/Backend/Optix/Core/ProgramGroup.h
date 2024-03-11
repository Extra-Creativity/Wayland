#pragma once
#include "HostUtils/CommonHeaders.h"

#include "ContextManager.h"
#include "Module.h"

#include <vector>

namespace Wayland::Optix
{

/// @brief Array of program group to be linked as pipeline and used SBT.
class ProgramGroupArray
{
    void AddProgramGroup_(const OptixProgramGroupDesc &, std::size_t);
    void GeneralAddHitProgramGroup_(auto &&, auto &&, auto &&);
    void CleanProgramGroups_() noexcept;

public:
    ProgramGroupArray() = default;
    ProgramGroupArray(std::size_t expectedNum)
    {
        programNames_.reserve(expectedNum);
        programGroups_.reserve(expectedNum);
    }

    ProgramGroupArray(const ProgramGroupArray &) = delete;
    ProgramGroupArray &operator=(const ProgramGroupArray &) = delete;

    ProgramGroupArray(ProgramGroupArray &&) noexcept = default;
    ProgramGroupArray &operator=(ProgramGroupArray &&another) noexcept
    {
        CleanProgramGroups_();
        programNames_ = std::move(another.programNames_);
        programGroups_ = std::move(another.programGroups_);
    }

    ~ProgramGroupArray() { CleanProgramGroups_(); }

    /// @brief Add raygen program to the array, with pure name.
    /// @param module module that the program is in; this should be guaranteed
    /// by the user.
    /// @param name name without prefix (i.e.__raygen__).
    /// @return changed array, so that setters can be chained.
    ProgramGroupArray &AddRaygenProgramGroup(const Module &module,
                                             std::string_view name);
    /// @brief Add hit program to the array, with pure name.
    /// @param module module that the program is in; this should be guaranteed
    /// by the user.
    /// @param names names without prefix, the order should be IS, AH and CH
    /// (i.e. __intersection__, __anyhit__, __closesthit__).
    /// @return changed array, so that setters can be chained.
    ProgramGroupArray &AddHitProgramGroup(
        const Module &module, const std::array<std::string_view, 3> &names);

    /// @brief Add hit program to the array, with pure name.
    /// @param modules array of modules that each program is in, the order
    /// should be IS, AH and CH; this should be guaranteed by the user.
    /// @param names names without prefix, the order should be IS, AH and CH
    /// (i.e. __intersection__, __anyhit__, __closesthit__).
    /// @return changed array, so that setters can be chained.
    ProgramGroupArray &AddHitProgramGroup(
        const std::array<const Module *, 3> &modules,
        const std::array<std::string_view, 3> &names);
    /// @brief Add miss program to the array, with pure name.
    /// @param module module that the program is in; this should be guaranteed
    /// by the user.
    /// @param name name without prefix (i.e.__miss__).
    /// @return changed array, so that setters can be chained.
    ProgramGroupArray &AddMissProgramGroup(const Module &module,
                                           std::string_view name);

    // For raw APIs, prefix should be added by the user.
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

} // namespace Wayland::Optix