#include "ContextManager.h"

#include "spdlog/spdlog.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <ranges>
#include <string_view>
#include <version>

thread_local ContextManager::ContextInfo *LocalContextSetter::currentContext_ =
    nullptr;

static void LogOptix(unsigned int level, const char *tag, const char *message,
                     [[maybe_unused]] void *cbdata)
{
    constexpr std::string_view fmt = "[{}]{}";
    switch (level)
    {
    case 1: // fatal
        SPDLOG_CRITICAL(fmt, tag, message);
        break;
    case 2: // error
        SPDLOG_ERROR(fmt, tag, message);
        break;
    case 3: // warning
        SPDLOG_WARN(fmt, tag, message);
        break;
    case 4: // print
        SPDLOG_INFO(fmt, tag, message);
        break;
    [[unlikely]] default:
        SPDLOG_CRITICAL("Unknown logging level {}.", level);
        break;
    }
}

static int InitAndGetDeviceCount()
{
    // Force to load devices.
    cudaFree(0);
    int count = 0;
    cudaGetDeviceCount(&count);

    [[maybe_unused]] static int _ = []() {
        HostUtils::CheckOptixError(optixInit());
        return 0;
    }();

    return count;
}

ContextManager::OptixDeviceContextWrapper::OptixDeviceContextWrapper(
    bool &success) noexcept
{
    OptixDeviceContextOptions options{};
    options.logCallbackLevel =
        std::clamp(spdlog::level::n_levels - spdlog::get_level(), 0, 4);
    options.logCallbackFunction = LogOptix;
#ifdef NEED_VALIDATION_MODE
    options.validationMode = OptixDeviceContextValidationMode::
        OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

    success = (optixDeviceContextCreate(CUcontext{ 0 }, &options, &context) ==
               OptixResult::OPTIX_SUCCESS);
}

// nullptr means a virtual global context in cuda, which cannot be freed.
ContextManager::OptixDeviceContextWrapper::~OptixDeviceContextWrapper()
{
    if (context)
    {
        HostUtils::CheckOptixError<HostUtils::OnlyLog>(
            optixDeviceContextDestroy(context));
    }
}

ContextManager::CUDAStreamWrapper::CUDAStreamWrapper(bool &success) noexcept
{
    success = (cudaStreamCreate(&stream) == cudaError::cudaSuccess);
}

// nullptr means a virtual global stream in cuda, which cannot be freed.
ContextManager::CUDAStreamWrapper::~CUDAStreamWrapper()
{
    if (stream)
    {
        HostUtils::CheckCUDAError<HostUtils::OnlyLog>(
            cudaStreamDestroy(stream));
    }
}

static std::optional<ContextManager::ContextInfo> CreateContext(
    int deviceIdx) noexcept
{
    using ContextInfo = ContextManager::ContextInfo;

    if (cudaDeviceProp deviceProperty;
        cudaGetDeviceProperties(&deviceProperty, deviceIdx) !=
        cudaError::cudaSuccess)
    {
        SPDLOG_WARN("Unable to get name of device #{}.", deviceIdx);
    }
    else
    {
        SPDLOG_INFO("Checking device #{}: {}...", deviceIdx,
                    deviceProperty.name);
    }

    int unifiedMemoryCheck = 0;
    if (cudaSetDevice(deviceIdx) != cudaError::cudaSuccess ||
        cudaDeviceGetAttribute(&unifiedMemoryCheck,
                               cudaDevAttrConcurrentManagedAccess,
                               deviceIdx) != cudaError::cudaSuccess)
    {
        SPDLOG_ERROR("Unable to initialize the current GPU.");
        return std::nullopt;
    }

    bool success;
    decltype(ContextInfo::context) context{ success };
    if (!success)
    {
        SPDLOG_ERROR("Unable to initialize optix context on the current GPU.");
        return std::nullopt;
    }

    decltype(ContextInfo::stream) stream{ success };
    if (!success)
    {
        SPDLOG_ERROR("Unable to create cuda stream on the current GPU.");
        return std::nullopt;
    }

    return ContextInfo{ .deviceID = deviceIdx,
                        .context = std::move(context),
                        .stream = std::move(stream),
                        .canAsyncAccessUnifiedMemory = unifiedMemoryCheck };
}

ContextManager::ContextManager()
{
    int count = InitAndGetDeviceCount();
    for (int deviceID = 0; deviceID < count; deviceID++)
    {
        if (auto result = CreateContext(deviceID); result)
            contexts_.push_back(std::move(*result));
    }

    CheckContextAvailable_();
}

ContextManager::ContextManager(std::span<const int> deviceIDs)
{
    int count = InitAndGetDeviceCount();
    for (auto deviceID : deviceIDs)
    {
        if (deviceID >= count || deviceID < 0)
        {
            SPDLOG_WARN("Totally only {} devices, id #{} is skipped.", count,
                        deviceID);
            continue;
        }
        if (auto result = CreateContext(deviceID); result)
            contexts_.push_back(std::move(*result));
    }
    CheckContextAvailable_();
}

void ContextManager::CheckContextAvailable_()
{
    HostUtils::CheckError(!contexts_.empty(), "No available cuda devices.",
                          "Unable to initialize optix context.");
#ifdef __cpp_lib_format_ranges
    SPDLOG_INFO("Optix initialized on devices: {}",
                contexts_ | std::views::transform([](const auto &context) {
                    return context.deviceID;
                }));
#else
    auto contextStr =
        contexts_ | std::views::transform([](const auto &context) {
            return std::to_string(context.deviceID);
        }) |
        std::views::join_with(',') | std::ranges::to<std::string>();
    SPDLOG_INFO("Optix initialized on devices: [{}]", std::move(contextStr));
#endif
    // set the current context as the first one by default.
    [[maybe_unused]] LocalContextSetter _{ *this, 0 };
}

void ContextManager::SetCachePath(const std::string &path)
{
    for (const auto &info : contexts_)
    {
        HostUtils::CheckOptixError(optixDeviceContextSetCacheLocation(
            info.context.context, path.c_str()));
    }
}