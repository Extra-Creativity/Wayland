#pragma once
#include "HostUtils/CommonHeaders.h"
#include "HostUtils/SpecialMacros.h"

#include "ProgramConfig.h"

namespace EasyRender::Optix
{

class ProgramGroupArray;

/// @brief Abstraction of OptixPipeline.
/// @details A pipeline needs a ProgramArray and PipelineConfig to set up the
/// stack and be built.
class Pipeline
{
public:
    /// @brief Construct a pipeline using the safest way (but may not be space
    /// efficient)
    /// @param arr ProgramGroupArray
    /// @param maxTraceDepth the max bounces of the raytracing.
    /// @param maxTraversableDepth the max depth of the traversable geometry; we
    /// provide it by .GetDepth() in all traverables directly, so you don't need
    /// to count it manually.
    /// @param maxCCDepth max calling depth of continuation callable.
    /// @param maxDCDepth max depth of direct callable.
    /// @param pipelineConfig
    Pipeline(
        const ProgramGroupArray &arr,
        decltype(OptixPipelineLinkOptions::maxTraceDepth) maxTraceDepth,
        unsigned int maxTraversableDepth, unsigned int maxCCDepth = 0,
        unsigned int maxDCDepth = 0,
        const PipelineConfig &pipelineConfig = PipelineConfig::GetDefaultRef());

    /// @brief you can also provide OptixStackSizes additionally to eliminate
    /// procedure of accumulation, but this may be rarely used.
    Pipeline(
        const ProgramGroupArray &arr,
        decltype(OptixPipelineLinkOptions::maxTraceDepth) maxTraceDepth,
        const OptixStackSizes &stackSize, unsigned int maxTraversableDepth,
        unsigned int maxCCDepth = 0, unsigned int maxDCDepth = 0,
        const PipelineConfig &pipelineConfig = PipelineConfig::GetDefaultRef());

    /// @brief Construct a pipeline using stack size designated by the user,
    /// which then allows more space-efficient way.
    /// @param arr ProgramGroupArray
    /// @param maxTraceDepth the max bounces of the raytracing.
    /// @param maxTraversableDepth the max depth of the traversable geometry; we
    /// provide it by .GetDepth() in all traverables directly, so you don't need
    /// to count it manually.
    /// @param directCallableStackSizeFromTraversal the stack size of direct
    /// callable from the traversable geometry.
    /// @param directCallableStackSizeFromState the stack size of direct
    /// callable from the state.
    /// @param continuationStackSize the stack size of continuation callable.
    /// @param pipelineConfig
    Pipeline(
        const ProgramGroupArray &arr,
        decltype(OptixPipelineLinkOptions::maxTraceDepth) maxTraceDepth,
        unsigned int maxTraversableDepth,
        unsigned int directCallableStackSizeFromTraversal,
        unsigned int directCallableStackSizeFromState,
        unsigned int continuationStackSize,
        const PipelineConfig &pipelineConfig = PipelineConfig::GetDefaultRef());

    auto GetHandle() const noexcept { return pipeline_; }

    Pipeline(const Pipeline &) = delete;
    Pipeline &operator=(const Pipeline &) = delete;
    Pipeline(Pipeline &&) noexcept
        : pipeline_{ std::exchange(pipeline_, nullptr) }
    {
    }
    Pipeline &operator=(Pipeline &&) noexcept
    {
        CleanPipeline_();
        pipeline_ = std::exchange(pipeline_, nullptr);
        return *this;
    }

    ~Pipeline();

private:
    void CleanPipeline_() noexcept;

    OptixPipeline pipeline_;
    // For safety, i.e. used to check whether the scene exceeds limit.
    unsigned int maxTraversableDepth_;
};

} // namespace EasyRender::Optix