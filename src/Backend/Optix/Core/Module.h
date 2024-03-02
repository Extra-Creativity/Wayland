#pragma once
#include "HostUtils/CommonHeaders.h"
#include "ProgramConfig.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace Wayland::Optix
{

class PipelineConfig;
class ProgramGroupArray;

/// @brief Abstraction of OptixModule.
class Module
{
    static std::string s_optixSDKPath;
    static std::vector<std::string> s_additionalArgs;

    static const inline std::vector<std::string> s_defaultCompileArgs{
        "-optix-ir",
        "-m64",
        "-rdc true",
        "--std c++17",
        "--use_fast_math",
        "--generate-line-info",
        R"(-Xcompiler "/utf-8")" // use utf-8 as encoding of source code.
    };

    void CreateModule_(std::string_view, const ModuleConfig &,
                       const PipelineConfig &);

public:
    /// @brief Create OptixModule with one file compiled by nvcc.
    /// @param fileName The file name of the source code; you can also pass e.g.
    /// *.cu, as long as nvcc can identify it.
    /// @param moduleConfig configuration of modules.
    /// @param pipelineConfig configuration of pipeline that the program will be
    /// linked into.
    /// @param compileOptions arguments need to be passed to nvcc.
    Module(
        std::string_view fileName,
        const ModuleConfig &moduleConfig = ModuleConfig::GetDefaultRef(),
        const PipelineConfig &pipelineConfig = PipelineConfig::GetDefaultRef(),
        const std::vector<std::string> &compileOptions = s_defaultCompileArgs);

    /// @brief Create OptixModule with files compiled by nvcc.
    /// @param fileNames a bunch of files; other parameters are same as the ctor
    /// of the single name.
    /// @note We don't use std::vector<std::string_view> in public APIs because
    /// it's easy for users to get it wrong; see
    /// https://stackoverflow.com/questions/64221633.
    Module(
        const std::vector<std::string> &fileNames,
        const ModuleConfig &moduleConfig = ModuleConfig::GetDefaultRef(),
        const PipelineConfig &pipelineConfig = PipelineConfig::GetDefaultRef(),
        const std::vector<std::string> &compileOptions = s_defaultCompileArgs);

    /// @brief For general case, the command needs to be customize, like
    /// generate a complex target by cmake.
    /// @param command customized command.
    /// @param targetPath path of result binary code, either optix-ir or ptx.
    /// @param moduleConfig
    /// @param pipelineConfig
    Module(
        const char *command, const char *targetPath,
        const ModuleConfig &moduleConfig = ModuleConfig::GetDefaultRef(),
        const PipelineConfig &pipelineConfig = PipelineConfig::GetDefaultRef());

    /// @brief to compile optix to binary code, optix SDK path should be set. If
    /// you only use customization command, then this can be omitted.
    template<typename T>
        requires std::is_assignable_v<std::string, T>
    static void SetOptixSDKPath(T &&initPath)
    {
        s_optixSDKPath = std::forward<T &&>(initPath);
    }

    /// @brief add additional arguments to nvcc.
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
    /// @brief Fill ProgramGroupArray with programs identified from files. For
    /// hitgroups, program with the same suffix will be in the same hit group.
    /// @param identifyFiles files that contains all programs.
    /// @param arr ProgramGroupArray that will be filled.
    /// @note + The program should be decorated with `extern "C" __global__ void
    /// xxx();` the number of blank characters doesn't matter.
    /// @note + It's the user who needs to guarantee that all programs of
    /// identifyFile appear in the module (i.e. in compiled files in ctor), and
    /// the raygen program only appears at once to be successfully linked as
    /// pipeline.
    /// @note + Files identified one by one, and programs are identified one by
    /// one; the only exception is that hitgroups will be pushed finally at
    /// once, the order will be same as the order of sequence of first suffix.
    /// @note For example, if you have two files:
    /// @note 1. t1.cu, define programs in the order of __raygen__A,
    /// __closesthit__B, __miss__C, __anyhit__D
    /// @note 2. t2.cu, define programs in the order of __anyhit__B,
    /// __closesthit__D, __miss__E
    /// @note Then the program array will be filled in the order of __raygen__A,
    /// __miss__C, __miss__E, and then all hitgroups:
    /// @note hitGroupB (i.e. __closesthit__B and __anyhit__B, because
    /// __closesthit__B appears at the first file, and as the first hit group
    /// program there), hitGroupD (i.e. __closesthit__D and __anyhit__D)
    /// @note + This function is only available when NEED_AUTO_PROGRAM_CONFIG is
    /// defined.
    void IdentifyPrograms(const std::vector<std::string> &identifyFiles,
                          ProgramGroupArray &arr);
#endif
};

} // namespace Wayland::Optix