#pragma once
#include "HostUtils/CommonHeaders.h"
#include "ProgramConfig.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

class PipelineConfig;
class ProgramGroupArray;

class Module
{
    static std::string s_optixSDKPath;
    static std::vector<std::string> s_additionalArgs;

    static const inline std::vector<std::string> s_defaultCompileArgs{
        "-optix-ir",
        "-m64",
        "-rdc true",
        "--use_fast_math",
        "--generate-line-info",
        R"(-Xcompiler "/utf-8")"
    };

    void CreateModule_(std::string_view, const ModuleConfig &,
                       const PipelineConfig &);

public:
    Module(
        std::string_view fileName,
        const ModuleConfig &moduleConfig = ModuleConfig::GetDefaultRef(),
        const PipelineConfig &pipelineConfig = PipelineConfig::GetDefaultRef(),
        const std::vector<std::string> &compileOptions = s_defaultCompileArgs);
    Module(
        const std::vector<std::string> &fileNames,
        const ModuleConfig &moduleConfig = ModuleConfig::GetDefaultRef(),
        const PipelineConfig &pipelineConfig = PipelineConfig::GetDefaultRef(),
        const std::vector<std::string> &compileOptions = s_defaultCompileArgs);
    Module(
        const char *command, const char *targetPath,
        const ModuleConfig &moduleConfig = ModuleConfig::GetDefaultRef(),
        const PipelineConfig &pipelineConfig = PipelineConfig::GetDefaultRef());

    template<typename T>
        requires std::is_assignable_v<std::string, T>
    static void SetOptixSDKPath(T &&initPath)
    {
        s_optixSDKPath = std::forward<T &&>(initPath);
    }

    template<typename T>
        requires std::is_constructible_v<std::string, T>
    static void AddArgs(T &&arg)
    {
        s_additionalArgs.emplace_back(std::forward<T &&>(arg));
    }

    auto GetHandle() const noexcept { return module_; }

    ~Module();

private:
    OptixModule module_;
#ifdef NEED_AUTO_PROGRAM_CONFIG
public:
    void IdentifyPrograms(const std::vector<std::string> &identifyFiles,
                          ProgramGroupArray &arr);
#endif
};