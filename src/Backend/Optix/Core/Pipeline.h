#pragma once
#include "HostUtils/CommonHeaders.h"
#include "HostUtils/SpecialMacros.h"

#include "ProgramConfig.h"

class ProgramGroupArray;

class Pipeline
{
public:
    Pipeline(
        const ProgramGroupArray &arr,
        decltype(OptixPipelineLinkOptions::maxTraceDepth) maxTraceDepth,
        unsigned int maxTraversableDepth, unsigned int maxCCDepth = 0,
        unsigned int maxDCDepth = 0,
        const PipelineConfig &pipelineConfig = PipelineConfig::GetDefaultRef());

    Pipeline(
        const ProgramGroupArray &arr,
        decltype(OptixPipelineLinkOptions::maxTraceDepth) maxTraceDepth,
        const OptixStackSizes &stackSize, unsigned int maxTraversableDepth,
        unsigned int maxCCDepth = 0, unsigned int maxDCDepth = 0,
        const PipelineConfig &pipelineConfig = PipelineConfig::GetDefaultRef());

    Pipeline(
        const ProgramGroupArray &arr,
        decltype(OptixPipelineLinkOptions::maxTraceDepth) maxTraceDepth,
        unsigned int maxTraversableDepth,
        unsigned int directCallableStackSizeFromTraversal,
        unsigned int directCallableStackSizeFromState,
        unsigned int continuationStackSize,
        const PipelineConfig &pipelineConfig = PipelineConfig::GetDefaultRef());

    auto GetHandle() const noexcept { return pipeline_; }

private:
    OptixPipeline pipeline_;
    // For safety, i.e. used to check whether the scene exceeds limit.
    unsigned int maxTraversableDepth_;
};