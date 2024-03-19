#pragma once

#include "HostUtils/DeviceAllocators.h"
#include "Traversable.h"
#include "glm/mat3x4.hpp"

#include <memory>
#include <span>

namespace Wayland::OptiX
{

class StaticTransform : public Traversable
{
public:
    StaticTransform(const glm::mat3x4 &transform,
                    const Traversable &childTraversable);

    void SetNewTransform(const glm::mat3x4 &transform);
    std::string DisplayInfo() const override { return "static transform"; }
    unsigned int GetDepth() const noexcept override
    {
        return UncheckedGetDepthForSingleChild_(childTraversablePtr_);
    }

private:
    const Traversable *childTraversablePtr_;
    Wayland::HostUtils::DeviceUniquePtr<OptixStaticTransform> buffer_;
    void SetUpTraversableHandle_(const OptixStaticTransform &transform);
};

template<typename Optix_TransformType>
class MotionTransformBase : public Traversable
{
public:
    unsigned int GetDepth() const noexcept override
    {
        return UncheckedGetDepthForSingleChild_(childTraversablePtr_);
    }

    MotionTransformBase(const Traversable &childTraversable)
        : childTraversablePtr_{ &childTraversable }
    {
    }

protected:
    // Not OptixMotionTransform: because it assumes that transform has varying
    // length, so its size cannot be represented by the sizeof().
    Wayland::HostUtils::DeviceUniquePtr<std::byte[]> buffer_;
    template<OptixTraversableType TypeEnum>
    void SetUpTraversableHandle_(std::span<const std::byte>);

private:
    const Traversable *childTraversablePtr_;
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

} // namespace Wayland::OptiX