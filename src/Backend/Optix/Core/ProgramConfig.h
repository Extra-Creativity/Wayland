#pragma once
#include "HostUtils/CommonHeaders.h"
#include "HostUtils/ErrorCheck.h"
#include "HostUtils/SpecialMacros.h"

#include "spdlog/spdlog.h"

#include <span>
#include <string_view>

class ModuleConfig
{
    OptixModuleCompileOptions option_;

    static const inline OptixModuleCompileOptions s_defaultModuleOptions{
        .maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        .optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT,
        .boundValues = nullptr,
        .numBoundValues = 0,
        .numPayloadTypes = 0,
        .payloadTypes = nullptr
    };

    constexpr ModuleConfig(const OptixModuleCompileOptions &init_option)
        : option_{ init_option }
    {
    }

public:
    ModuleConfig() = default;
    static const auto &GetDefaultRef() noexcept
    {
        static ModuleConfig s_defaultConfig{ s_defaultModuleOptions };
        return s_defaultConfig;
    }
    static auto GetDefault() noexcept { return GetDefaultRef(); }

    const auto &GetRawOptions() const noexcept { return option_; }
    auto &SetPayload(std::span<OptixPayloadType> payloads)
    {
        using PayloadTypeNumType = decltype(option_.numPayloadTypes);
        HostUtils::CheckError(
            payloads.size() <= (std::numeric_limits<PayloadTypeNumType>::max)(),
            "Too many payloads");
        option_.payloadTypes = payloads.data();
        option_.numPayloadTypes =
            static_cast<PayloadTypeNumType>(payloads.size());
        return *this;
    }

    auto &SetBoundValue(
        std::span<OptixModuleCompileBoundValueEntry> boundValues)
    {
        using BoundValueNumType = decltype(option_.numBoundValues);
        HostUtils::CheckError(
            boundValues.size() <=
                (std::numeric_limits<BoundValueNumType>::max)(),
            "Too many bound values");
        option_.boundValues = boundValues.data();
        option_.numBoundValues =
            static_cast<BoundValueNumType>(boundValues.size());
        return *this;
    }

    DEFINE_SIMPLE_CHAINED_SETTER(MaxRegisterCount, option_.maxRegisterCount);
    DEFINE_SIMPLE_CHAINED_SETTER(OptimizeLevel, option_.optLevel);
    DEFINE_SIMPLE_CHAINED_SETTER(DebugLevel, option_.debugLevel);
};

class PipelineConfig
{
    OptixPipelineCompileOptions option_;

    static const inline OptixPipelineCompileOptions s_defaultModuleOptions
    {
        .usesMotionBlur = false,
        .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
        .numPayloadValues = 0, .numAttributeValues = 0,
        .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
        .pipelineLaunchParamsVariableName = "param",
        .usesPrimitiveTypeFlags = 0,
#if OPTIX_VERSION >= 8000
        .allowOpacityMicromaps = 0
#endif
    };

    constexpr PipelineConfig(const OptixPipelineCompileOptions &init_option)
        : option_{ init_option }
    {
    }

public:
    PipelineConfig() = default;
    static const auto &GetDefaultRef() noexcept
    {
        static PipelineConfig s_defaultConfig{ s_defaultModuleOptions };
        return s_defaultConfig;
    }
    static auto GetDefault() noexcept { return GetDefaultRef(); }
    const auto &GetRawOptions() const noexcept { return option_; }

    auto &SetSingleGASFlag()
    {
        option_.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        return *this;
    }

    auto &SetSingleIASToGASFlag()
    {
        option_.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        return *this;
    }

    DEFINE_SIMPLE_CHAINED_SETTER(MotionBlur, option_.usesMotionBlur);
    DEFINE_SIMPLE_CHAINED_SETTER(NumPayloadValues, option_.numPayloadValues);
    DEFINE_SIMPLE_CHAINED_SETTER(NumAttributeValues,
                                 option_.numAttributeValues);
    DEFINE_SIMPLE_CHAINED_SETTER(ExceptionFlags, option_.exceptionFlags);
    DEFINE_SIMPLE_CHAINED_SETTER(PrimitiveTypeFlags,
                                 option_.usesPrimitiveTypeFlags);
#if OPTIX_VERSION >= 8000
    DEFINE_SIMPLE_CHAINED_SETTER(AllowOpacityMicroMaps,
                                 option_.allowOpacityMicromaps);
#endif
};

inline void LogProcedureInfo(std::size_t logStringSize,
                             const std::size_t logMaxSize, const char *logPtr)
{
    logStringSize--;
    if (logStringSize > logMaxSize)
        SPDLOG_WARN("[Truncated Log]: {}",
                    std::string_view{ logPtr, logMaxSize });
    else if (logStringSize != 0) // i.e. not only a null termination.
        SPDLOG_INFO("{}", std::string_view{ logPtr, logStringSize });
}