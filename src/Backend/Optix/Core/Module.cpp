#include "Module.h"
#include "ContextManager.h"
#include "HostUtils/ErrorCheck.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <format>
#include <fstream>
#include <random>
#include <ranges>
#include <sstream>
#include <thread>

#include "spdlog/spdlog.h"

using namespace EasyRender;

namespace stdv = std::views;
namespace stdr = std::ranges;

/// @brief Create a temporary directory to store modules.
/// @details The directory is created when the program is initialized, and the
/// name is `./tmp/{date}-{time}-{subsecond}/bin`. When the program exits
/// normally (i.e. not quited by exception or abort), then all contents and the
/// directory will be removed.
struct TemporaryDirectory
{
    std::filesystem::path dir;
    TemporaryDirectory()
    {
        using namespace std::literals;
        namespace stdc = std::chrono;

        std::filesystem::path createPath;
        constexpr std::size_t maxTryTimes = 10;
        // create the directory by timestamp, try multiple times to prevent
        // different processes writing in the same directory.
        for (std::size_t tryCnt = 0; tryCnt < maxTryTimes; tryCnt++)
        {
            auto currTime =
                stdc::current_zone()->to_local(stdc::system_clock::now());

            auto daysTime = stdc::floor<stdc::days>(currTime);
            stdc::year_month_day date{ daysTime };
            stdc::hh_mm_ss hmsTime{ currTime - daysTime };

            // No trailing /, see
            // https://developercommunity.visualstudio.com/t/278829.
            createPath = std::format("./tmp/{:%F}-{:%H-%M-%S}-{}/bin", date,
                                     hmsTime, hmsTime.subseconds().count());

            if (std::error_code err;
                std::filesystem::create_directories(createPath, err))
            {
                dir = std::move(createPath);
                return;
            }
            else if (err) // Unknown error, terminate the program...
            {
                SPDLOG_CRITICAL(
                    "For path {}, {}",
                    reinterpret_cast<char *>(createPath.u8string().data()),
                    err.message());
                break;
            }
            // Wait for random time, we just make it simple.
            std::this_thread::sleep_for(100ms +
                                        (std::random_device{}() % 128) * 1ms);
        }
        SPDLOG_CRITICAL("Unable to create directory to store modules.");
        std::terminate(); // fatal error, unable to store modules.
    }
    ~TemporaryDirectory()
    {
        if (!dir.empty())
            std::filesystem::remove_all(dir.parent_path());
    }
};

static const TemporaryDirectory s_binSaveDir{};
static std::atomic<std::uint64_t> s_uid{}; // used as id of binary module code.
static constexpr std::size_t s_moduleLogSize = 1024;
static char s_moduleLog[s_moduleLogSize];

static std::string GetNewSavePath()
{
    return std::format("{}/module-{}.optixir", s_binSaveDir.dir.string(),
                       s_uid.fetch_add(1, std::memory_order_relaxed));
}

static std::string JoinArgs(const std::vector<std::string> &args)
{
    return args | stdv::join_with(' ') | stdr::to<std::string>();
}

namespace EasyRender::Optix
{

std::string Module::s_optixSDKPath{};
std::vector<std::string> Module::s_additionalArgs;

static auto GetOutputFile(const char *command, const char *path)
{
    SPDLOG_INFO("Executing command {}", command);
    // Generate the output file by the command
    std::system(command);

    // And read it into the stream.
    std::ostringstream str;
    std::ifstream fin{ path, std::ios::binary };
    HostUtils::CheckError(fin.is_open(),
                          "Fail to generate optix binary from cuda files.");
    str << fin.rdbuf();
    HostUtils::CheckError(str.good(), "Fail to read optix binary files.");
    return str;
}

#if OPTIX_VERSION >= 77000
#define optixModuleCreate optixModuleCreate
#else
#define optixModuleCreate optixModuleCreateFromPTX
#endif

/// @brief It should be called as the final procedure, otherwise exception may
///        leave module_ not destroyed.
/// @param view byte view of binary code.
/// @param moduleConfig
/// @param pipelineConfig
void Module::CreateModule_(std::string_view view,
                           const ModuleConfig &moduleConfig,
                           const PipelineConfig &pipelineConfig)
{
    size_t logStringSize = s_moduleLogSize;
    auto result = optixModuleCreate(
        LocalContextSetter::GetCurrentOptixContext(),
        &moduleConfig.GetRawOptions(), &pipelineConfig.GetRawOptions(),
        view.data(), view.size(), s_moduleLog, &logStringSize, &module_);
    LogProcedureInfo(logStringSize, s_moduleLogSize, s_moduleLog);
    HostUtils::CheckOptixError(result);
}

Module::Module(std::string_view fileName, const ModuleConfig &moduleConfig,
               const PipelineConfig &pipelineConfig,
               const std::vector<std::string> &compileOptions)
{
    auto outPath = GetNewSavePath();
    std::string command = std::format(
        R"(nvcc -I"{}" {} {} "{}" -o "{}")", s_optixSDKPath,
        stdv::join_with(compileOptions, ' ') | stdr::to<std::string>(),
        JoinArgs(s_additionalArgs), fileName, outPath);
    auto fileContents = GetOutputFile(command.c_str(), outPath.c_str());
    CreateModule_(fileContents.view(), moduleConfig, pipelineConfig);
}

Module::Module(const std::vector<std::string> &fileNames,
               const ModuleConfig &moduleConfig,
               const PipelineConfig &pipelineConfig,
               const std::vector<std::string> &compileOptions)
{
    auto outPath = GetNewSavePath();
    std::string command = std::format(
        R"(nvcc -I"{}" {} {} "{}" -o "{}")", s_optixSDKPath,
        stdv::join_with(compileOptions, ' ') | stdr::to<std::string>(),
        JoinArgs(s_additionalArgs),
        stdv::join_with(fileNames, ' ') | stdr::to<std::string>(), outPath);
    auto fileContents = GetOutputFile(command.c_str(), outPath.c_str());
    CreateModule_(fileContents.view(), moduleConfig, pipelineConfig);
}

Module::Module(const char *command, const char *targetPath,
               const ModuleConfig &moduleConfig,
               const PipelineConfig &pipelineConfig)
{
    auto fileContents = GetOutputFile(command, targetPath);
    CreateModule_(fileContents.view(), moduleConfig, pipelineConfig);
}

Module::~Module()
{
    HostUtils::CheckOptixError<HostUtils::OnlyLog>(optixModuleDestroy(module_));
}

} // namespace EasyRender::Optix