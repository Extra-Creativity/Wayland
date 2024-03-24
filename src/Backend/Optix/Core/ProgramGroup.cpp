#include "ProgramGroup.h"
#include "HostUtils/ErrorCheck.h"

#include <ranges>

constexpr std::size_t s_programLogSize = 1024;
static char s_programLog[s_programLogSize];

using namespace EasyRender;

namespace EasyRender::Optix
{

/// @brief Safely add program group, so that when the insertion fails,
/// programNames can be correctly popped up to release memory timely. This
/// achieves strong exception guarantee.
/// @param desc program group description that is used to create a program
/// group.
/// @param popNum when insertion fails, how many names should be removed.
void ProgramGroupArray::AddProgramGroup_(const OptixProgramGroupDesc &desc,
                                         std::size_t popNum = 1)
{
    OptixProgramGroup group = nullptr;
    OptixProgramGroupOptions pgOptions{};
    std::size_t logStringSize = s_programLogSize;
    try
    {
        auto result = optixProgramGroupCreate(
            LocalContextSetter::GetCurrentOptixContext(), &desc, 1, &pgOptions,
            s_programLog, &logStringSize, &group);
        LogProcedureInfo(logStringSize, s_programLogSize, s_programLog);
        HostUtils::CheckOptixError(result);
        SPDLOG_INFO(
            "Add new program group: {} - {}", programGroups_.size(), [&]() {
                return std::ranges::subrange{ programNames_.end() - popNum,
                                              programNames_.end() } |
                       std::views::join_with(',') |
                       std::ranges::to<std::string>();
            }());
        programGroups_.push_back(group);
    }
    catch (...)
    {
        SPDLOG_ERROR("Cannot create raygen program, trying to destroy it...");
        programNames_.erase(programNames_.end() - popNum, programNames_.end());
        if (group)
            HostUtils::CheckOptixError(optixProgramGroupDestroy(group));
        throw;
    }
}

ProgramGroupArray &ProgramGroupArray::AddRawRaygenProgramGroup(
    const Module &module, std::string initRawName)
{
    auto &rawName = programNames_.emplace_back(std::move(initRawName));
    OptixProgramGroupDesc desc{
        .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        .raygen = { .module = module.GetHandle(),
                    .entryFunctionName = rawName.c_str() },
    };
    AddProgramGroup_(desc);
    return *this;
}

ProgramGroupArray &ProgramGroupArray::AddRaygenProgramGroup(
    const Module &module, std::string_view name)
{
    return AddRawRaygenProgramGroup(module, "__raygen__" + std::string{ name });
}

using FillHandleType = void (*)(const Module &, const char *,
                                OptixProgramGroupDesc &);
static FillHandleType s_fillHandles[3]{
    [](const Module &module, const char *name, OptixProgramGroupDesc &desc) {
        desc.hitgroup.moduleIS = module.GetHandle();
        desc.hitgroup.entryFunctionNameIS = name;
    },
    [](const Module &module, const char *name, OptixProgramGroupDesc &desc) {
        desc.hitgroup.moduleAH = module.GetHandle();
        desc.hitgroup.entryFunctionNameAH = name;
    },
    [](const Module &module, const char *name, OptixProgramGroupDesc &desc) {
        desc.hitgroup.moduleCH = module.GetHandle();
        desc.hitgroup.entryFunctionNameCH = name;
    }
};

/// @brief Add hit program group safely (i.e. strong exception guarantee). We
/// don't add strict constraints since it's a private method.
/// @param module either a single module, or container of modules.
/// @param names array of program names of hit group.
/// @param handle a function that accepts an index (0 means IS, 1 means AH, 2
/// means CH, same as regulations in public APIs) and names (i.e. the provided
/// parameter here), and returns a raw name of hit program. The returned name
/// will be pushed into the program names.
void ProgramGroupArray::GeneralAddHitProgramGroup_(auto &&module, auto &&names,
                                                   auto &&handle)
{
    constexpr bool isSingleModule =
        std::is_same_v<std::remove_cvref_t<decltype(module)>, Module>;

    std::size_t validNum = 0;
    OptixProgramGroupDesc desc{};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    try
    {
        for (std::size_t i = 0; i < 3; i++)
        {
            if (names[i].empty())
                continue;
            auto &rawName = programNames_.emplace_back(handle(i, names));
            validNum++;
            if constexpr (isSingleModule)
                s_fillHandles[i](module, rawName.c_str(), desc);
            else
                s_fillHandles[i](*(module[i]), rawName.c_str(), desc);
        }
        if (validNum == 0)
        {
            SPDLOG_WARN("No valid hit group name.");
            return;
        }
    }
    catch (...)
    {
        programNames_.erase(programNames_.end() - validNum,
                            programNames_.end());
        throw;
    }
    AddProgramGroup_(desc, validNum);
}

ProgramGroupArray &ProgramGroupArray::AddRawHitProgramGroup(
    const Module &module, std::array<std::string, 3> &rawNames)
{
    GeneralAddHitProgramGroup_(module, rawNames, [](auto idx, auto &&names) {
        return std::move(names[idx]);
    });
    return *this;
}

ProgramGroupArray &ProgramGroupArray::AddRawHitProgramGroup(
    const std::array<const Module *, 3> &modules,
    std::array<std::string, 3> &rawNames)
{
    GeneralAddHitProgramGroup_(modules, rawNames, [](auto idx, auto &&names) {
        return std::move(names[idx]);
    });
    return *this;
}

constexpr const std::array<std::string_view, 3> s_hitProgramPrefixes{
    "__intersection__", "__anyhit__", "__closesthit__"
};

ProgramGroupArray &ProgramGroupArray::AddHitProgramGroup(
    const Module &module, const std::array<std::string_view, 3> &names)
{
    GeneralAddHitProgramGroup_(module, names, [](auto idx, auto &&names) {
        return std::string{ s_hitProgramPrefixes[idx] } +
               std::string{ names[idx] };
    });
    return *this;
}

ProgramGroupArray &ProgramGroupArray::AddHitProgramGroup(
    const std::array<const Module *, 3> &modules,
    const std::array<std::string_view, 3> &names)
{
    GeneralAddHitProgramGroup_(modules, names, [](auto idx, auto &&names) {
        return std::string{ s_hitProgramPrefixes[idx] } +
               std::string{ names[idx] };
    });
    return *this;
}

ProgramGroupArray &ProgramGroupArray::AddRawMissProgramGroup(
    const Module &module, std::string initRawName)
{
    auto &rawName = programNames_.emplace_back(std::move(initRawName));
    OptixProgramGroupDesc desc{
        .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
        .miss = { .module = module.GetHandle(),
                  .entryFunctionName = rawName.c_str() },
    };
    AddProgramGroup_(desc);
    return *this;
}

ProgramGroupArray &ProgramGroupArray::AddMissProgramGroup(const Module &module,
                                                          std::string_view name)
{
    return AddRawMissProgramGroup(module, "__miss__" + std::string{ name });
}

} // namespace EasyRender::Optix