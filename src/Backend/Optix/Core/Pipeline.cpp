#include "optix_stack_size.h"

#include "HostUtils/SpecialMacros.h"
#include "Pipeline.h"
#include "ProgramGroup.h"
#include "spdlog/spdlog.h"

static constexpr std::size_t s_pipelineLogSize = 1024;
static char s_pipelineLog[s_pipelineLogSize];

using namespace EasyRender;

namespace EasyRender::Optix
{

// RAII to prevent pipeline leak. Its Release() is supposed to be
// called before it's destructed.
class PipelineWrapper
{
    OptixPipeline pipeline_ = nullptr;

public:
    auto GetRaw() const noexcept { return pipeline_; }
    auto Release() noexcept { return std::exchange(pipeline_, nullptr); }
    PipelineWrapper(
        const ProgramGroupArray &arr,
        decltype(OptixPipelineLinkOptions::maxTraceDepth) maxTraceDepth,
        const PipelineConfig &pipelineConfig)
    {
        const auto &programGroups = arr.GetHandleArr();
        auto size = programGroups.size();
        HostUtils::CheckError(
            HostUtils::CheckInRange<unsigned int>(size),
            "Too many program groups to construct a pipeline.");

        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = maxTraceDepth;

        size_t logStringSize = s_pipelineLogSize;

        auto result = optixPipelineCreate(
            LocalContextSetter::GetCurrentOptixContext(),
            &pipelineConfig.GetRawOptions(), &pipelineLinkOptions,
            programGroups.data(), static_cast<unsigned int>(size),
            s_pipelineLog, &logStringSize, &pipeline_);
        LogProcedureInfo(logStringSize, s_pipelineLogSize, s_pipelineLog);
        HostUtils::CheckOptixError(result);
    }

    DISABLE_ALL_SPECIAL_METHODS(PipelineWrapper)

    ~PipelineWrapper()
    {
        if (pipeline_)
        {
            SPDLOG_ERROR("Unable to calculate stack size.");
            HostUtils::CheckOptixError<HostUtils::OnlyLog>(
                optixPipelineDestroy(pipeline_));
        }
    }
};

static inline OptixStackSizes AccumulateStackSize(const auto &programGroups,
                                                  OptixPipeline pipeline)
{
    OptixStackSizes stackSize{};
    for (const auto &programGroup : programGroups)
    {
        HostUtils::CheckOptixError(
            optixUtilAccumulateStackSizes(programGroup, &stackSize, pipeline));
    }
    return stackSize;
}

static void SetStackSize(
    const OptixStackSizes &stackSize, const PipelineWrapper &pipeline,
    decltype(OptixPipelineLinkOptions::maxTraceDepth) maxTraceDepth,
    unsigned int maxTraversableDepth, unsigned int maxCCDepth,
    unsigned int maxDCDepth)
{
    unsigned int directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState, continuationStackSize;
    HostUtils::CheckOptixError(optixUtilComputeStackSizes(
        &stackSize, maxTraceDepth, maxCCDepth, maxDCDepth,
        &directCallableStackSizeFromTraversal,
        &directCallableStackSizeFromState, &continuationStackSize));
    HostUtils::CheckOptixError(optixPipelineSetStackSize(
        pipeline.GetRaw(), directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState, continuationStackSize,
        maxTraversableDepth));
}

Pipeline::Pipeline(
    const ProgramGroupArray &arr,
    decltype(OptixPipelineLinkOptions::maxTraceDepth) maxTraceDepth,
    unsigned int maxTraversableDepth, unsigned int maxCCDepth,
    unsigned int maxDCDepth, const PipelineConfig &pipelineConfig)
    : maxTraversableDepth_{ maxTraversableDepth }
{
    PipelineWrapper pipeline{ arr, maxTraceDepth, pipelineConfig };
    OptixStackSizes stackSize =
        AccumulateStackSize(arr.GetHandleArr(), pipeline.GetRaw());
    SetStackSize(stackSize, pipeline, maxTraceDepth, maxTraversableDepth,
                 maxCCDepth, maxDCDepth);
    pipeline_ = pipeline.Release();
}

Pipeline::Pipeline(
    const ProgramGroupArray &arr,
    decltype(OptixPipelineLinkOptions::maxTraceDepth) maxTraceDepth,
    const OptixStackSizes &stackSize, unsigned int maxTraversableDepth,
    unsigned int maxCCDepth, unsigned int maxDCDepth,
    const PipelineConfig &pipelineConfig)
    : maxTraversableDepth_{ maxTraversableDepth }
{
    PipelineWrapper pipeline{ arr, maxTraceDepth, pipelineConfig };
    SetStackSize(stackSize, pipeline, maxTraceDepth, maxTraversableDepth,
                 maxCCDepth, maxDCDepth);
    pipeline_ = pipeline.Release();
}

Pipeline::Pipeline(
    const ProgramGroupArray &arr,
    decltype(OptixPipelineLinkOptions::maxTraceDepth) maxTraceDepth,
    unsigned int maxTraversableDepth,
    unsigned int directCallableStackSizeFromTraversal,
    unsigned int directCallableStackSizeFromState,
    unsigned int continuationStackSize, const PipelineConfig &pipelineConfig)
    : maxTraversableDepth_{ maxTraversableDepth }
{
    PipelineWrapper pipeline{ arr, maxTraceDepth, pipelineConfig };
    HostUtils::CheckOptixError(optixPipelineSetStackSize(
        pipeline.GetRaw(), directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState, continuationStackSize,
        maxTraversableDepth));
    pipeline_ = pipeline.Release();
}

void Pipeline::CleanPipeline_() noexcept
{
    if (pipeline_)
    {
        HostUtils::CheckOptixError<HostUtils::OnlyLog>(
            optixPipelineDestroy(pipeline_));
    }
}

Pipeline::~Pipeline()
{
    CleanPipeline_();
}

} // namespace EasyRender::Optix