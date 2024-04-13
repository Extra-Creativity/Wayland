#pragma once

#include "HostUtils/DeviceAllocators.h"
#include "Traversable.h"
#include "glm/mat3x4.hpp"

#include <memory>
#include <span>

namespace EasyRender::Optix
{

class TransformBase : public Traversable
{
public:
    TransformBase(const Traversable &childTraversable)
        : childTraversablePtr_{ &childTraversable }
    {
    }

    unsigned int GetDepth() const noexcept override
    {
        return UncheckedGetDepthForSingleChild_(childTraversablePtr_);
    }

    void FillSBT(unsigned int rayTypeNum,
                 SBTHitRecordBufferProxy &buffer) const override
    {
        return childTraversablePtr_->FillSBT(rayTypeNum, buffer);
    }

private:
    const Traversable *childTraversablePtr_;
};

class StaticTransform : public TransformBase
{
public:
    StaticTransform(const glm::mat3x4 &transform,
                    const Traversable &childTraversable);

    void SetNewTransform(const glm::mat3x4 &transform);
    std::string DisplayInfo() const override { return "static transform"; }

private:
    const Traversable *childTraversablePtr_;
    EasyRender::HostUtils::DeviceUniquePtr<OptixStaticTransform> buffer_;
    void SetUpTraversableHandle_(const OptixStaticTransform &transform);
};

template<typename Optix_TransformType>
class MotionTransformBase : public TransformBase
{
public:
    MotionTransformBase(const Traversable &childTraversable)
        : TransformBase{ childTraversable }
    {
    }

protected:
    // Not OptixMotionTransform: because it assumes that transform has varying
    // length, so its size cannot be represented by the sizeof().
    EasyRender::HostUtils::DeviceUniquePtr<std::byte[]> buffer_;
    template<OptixTraversableType TypeEnum>
    void SetUpTraversableHandle_(std::span<const std::byte>);
};

class MatrixMotionTransform
    : public MotionTransformBase<OptixMatrixMotionTransform>
{
public:
    std::string DisplayInfo() const override
    {
        return "matrix motion transform";
    }

    MatrixMotionTransform(std::span<const glm::mat3x4> transform,
                          const Traversable &childTraversable, float tBegin,
                          float tEnd);
    MatrixMotionTransform(std::span<const glm::mat3x4> transform,
                          const Traversable &childTraversable, float tBegin,
                          float tEnd, bool beginVanish, bool endVanish);
};

class SRTMotionTransform : public MotionTransformBase<OptixSRTMotionTransform>
{
public:
    std::string DisplayInfo() const override { return "SRT motion transform"; }

    SRTMotionTransform(std::span<const OptixSRTData> transforms,
                       const Traversable &childTraversable, float tBegin,
                       float tEnd);
    SRTMotionTransform(std::span<const OptixSRTData> transforms,
                       const Traversable &childTraversable, float tBegin,
                       float tEnd, bool beginVanish, bool endVanish);
};

} // namespace EasyRender::Optix