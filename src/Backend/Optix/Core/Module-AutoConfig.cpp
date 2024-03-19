#ifdef NEED_AUTO_PROGRAM_CONFIG
#include "Module.h"
#include "ProgramGroup.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <type_traits>

#define FUNCTION_SIG_HEAD_REGEX \
    R"++(extern(?:\s+?)"C"(?:\s+?)__global__(?:\s+?)void(?:\s+?))++"
// Function name of programs must begin with __.
// Name is captured to judge whether it's really a program; though this
// can be part of regex, it would make it too long and possibly not
// efficient (NOT profiled).
#define FUNCTION_SIG_NAME_REGEX R"((__[_0-9a-zA-Z\-]+?))"
#define FUNCTION_SIG_END_REGEX R"((?:\s*?)\(\))"

#ifdef NEED_RE2
#include "re2/re2.h"
static re2::RE2 s_funcRegex{
    FUNCTION_SIG_HEAD_REGEX FUNCTION_SIG_NAME_REGEX FUNCTION_SIG_END_REGEX
};
#else
#include <regex>
static std::regex s_funcRegex{
    FUNCTION_SIG_HEAD_REGEX FUNCTION_SIG_NAME_REGEX FUNCTION_SIG_END_REGEX
};
#endif

enum ProgramType
{
    Raygen,
    Intersection,
    AnyHit,
    ClosestHit,
    Miss
};

static constexpr std::array<std::string_view, 8> s_programPrefix{
    "raygen__",
    "intersection__",
    "anyhit__",
    "closesthit__",
    "miss__",
    "direct_callable__",
    "continuation_callable__",
    "exception__"
};

static_assert(s_programPrefix[Raygen] == "raygen__");
static_assert(s_programPrefix[Intersection] == "intersection__");
static_assert(s_programPrefix[AnyHit] == "anyhit__");
static_assert(s_programPrefix[ClosestHit] == "closesthit__");
static_assert(s_programPrefix[Miss] == "miss__");

// This guarantee is used by ClassifyProgram.
static_assert(AnyHit - Intersection == 1 && ClosestHit - AnyHit == 1);

struct HitgroupInfo
{
    HitgroupInfo(std::string_view init_suffix) : suffix{ init_suffix } {}

    std::string suffix;
    std::array<int, 3> infoExists{};

    bool operator==(std::string_view name) const noexcept
    {
        return name.ends_with(suffix);
    }
};

namespace Wayland::OptiX
{

static void ClassifyProgram(std::string_view funcName,
                            std::vector<HitgroupInfo> &hitGroupInfos,
                            const Module &module, ProgramGroupArray &arr)
{
    // strip out the prefix __.
    constexpr int commonPrefixLen = 2;

    auto it = std::ranges::find_if(
        s_programPrefix,
        [name = funcName.substr(commonPrefixLen)](std::string_view prefix) {
            return name.starts_with(prefix);
        });
    if (it == s_programPrefix.end())
        return;
    SPDLOG_INFO("Find program {}", funcName);

    auto type = it - s_programPrefix.begin();
    switch (type)
    {
    case Raygen:
        arr.AddRawRaygenProgramGroup(module, std::string{ funcName });
        return;
    case Intersection:
        [[fallthrough]];
    case AnyHit:
        [[fallthrough]];
    case ClosestHit: {
        // why not std::ranges::find: we don't need operator<=> of HitGroupInfo.
        auto pos =
            std::find(hitGroupInfos.begin(), hitGroupInfos.end(), funcName);
        auto funcNameSuffix =
            funcName.substr(commonPrefixLen + s_programPrefix[type].size());
        auto &info = pos == hitGroupInfos.end()
                         ? hitGroupInfos.emplace_back(funcNameSuffix)
                         : *pos;
        info.infoExists[type - Intersection] = 1;
        return;
    }
    case Miss:
        arr.AddRawMissProgramGroup(module, std::string{ funcName });
        return;
    default: // TODO: things like exception program.
        SPDLOG_WARN("Currently other program types isn't supported, so program "
                    "{} will be omitted.",
                    funcName);
        return;
    }
}

void Module::IdentifyPrograms(const std::vector<std::string> &identifyFiles,
                              ProgramGroupArray &arr)
{
    std::vector<HitgroupInfo> hitGroupInfos;
    std::ostringstream fileContent;
    for (const auto &file : identifyFiles)
    {
        fileContent << std::ifstream{ file }.rdbuf();
        auto pos = fileContent.tellp();
        if (pos == -1)
        {
            SPDLOG_WARN("Unable to read file {}", file);
            continue;
        }

#ifdef NEED_RE2
        re2::StringPiece view{ fileContent.view().data(), (size_t)pos };
        re2::StringPiece matchedPart;
        while (re2::RE2::FindAndConsume(&view, s_funcRegex, &matchedPart))
            ClassifyProgram({ matchedPart.data(), matchedPart.size() },
                            hitGroupInfos, *this, arr);
#else
        auto beginPtr = fileContent.view().data();
        for (std::cregex_iterator it{ beginPtr, beginPtr + (size_t)pos,
                                      s_funcRegex };
             it != std::cregex_iterator{}; it++)
        {
            auto &matchedResult = (*it)[1];
            ClassifyProgram({ matchedResult.first, matchedResult.second },
                            hitGroupInfos, *this, arr);
        }
#endif

        fileContent.seekp(0); // restore the file content.
        // Note: we don't use fileContent.str("") because it will release the
        // buffer and allocate again, which is unnecessary.
    }
    // Add all hitgroups to program array.
    std::array<std::string, 3> hitGroupNames;
    for (auto &info : hitGroupInfos)
    {
        for (std::size_t idx = 0; idx < hitGroupNames.size(); idx++)
        {
            auto &name = hitGroupNames[idx];
            if (info.infoExists[idx])
                name = "__" +
                       std::string{ s_programPrefix[idx + Intersection] } +
                       info.suffix;
            else
                name.clear();
        }
        arr.AddRawHitProgramGroup(*this, hitGroupNames);
    }
    return;
}

} // namespace Wayland::OptiX
#endif