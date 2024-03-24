#include "Transform.h"
#include "ContextManager.h"
#include "HostUtils/AlignedByteBuffer.h"
#include "HostUtils/ErrorCheck.h"

#include "glm/gtc/type_ptr.hpp"

#include <algorithm>
#include <limits>

using namespace EasyRender;

static constexpr auto cs_singleMatrixTransformSize = 12;
static constexpr auto cs_singleMatrixTransformBytesNum =
    cs_singleMatrixTransformSize * sizeof(float);

namespace EasyRender::Optix
{

static void SetTransformOfStaticTransform(OptixStaticTransform &result,
                                          const glm::mat3x4 &trans)
{
    auto ptr = glm::value_ptr(trans);
    std::copy(ptr, ptr + cs_singleMatrixTransformSize, result.transform);

    auto invTrans = glm::inverse(glm::mat4{ trans });
    assert(glm::all(
        glm::epsilonEqual(invTrans[3], glm::vec4{ 0, 0, 0, 1 }, 1e-5f)));

    auto invPtr = glm::value_ptr(invTrans);
    std::copy(invPtr, invPtr + cs_singleMatrixTransformSize,
              result.invTransform);
}

StaticTransform::StaticTransform(const glm::mat3x4 &init_trans,
                                 const Traversable &childTraversable)
    : childTraversablePtr_{ &childTraversable }
{
    OptixStaticTransform cpuTransform{ .child = childTraversable.GetHandle() };
    SetTransformOfStaticTransform(cpuTransform, init_trans);
    SetUpTraversableHandle_(cpuTransform);
}

void StaticTransform::SetNewTransform(const glm::mat3x4 &trans)
{
    OptixStaticTransform cpuTransform;
    SetTransformOfStaticTransform(cpuTransform, trans);
    HostUtils::CheckCUDAError(
        cudaMemcpy(buffer_->transform, &cpuTransform.transform,
                   sizeof(OptixStaticTransform) -
                       offsetof(OptixStaticTransform, transform),
                   cudaMemcpyHostToDevice));
}

void StaticTransform::SetUpTraversableHandle_(
    const OptixStaticTransform &cpuTransform)
{
    buffer_ = HostUtils::DeviceMakeUnique<OptixStaticTransform>(cpuTransform);
    HostUtils::CheckOptixError(optixConvertPointerToTraversableHandle(
        LocalContextSetter::GetCurrentOptixContext(),
        HostUtils::ToDriverPointer(buffer_.get()),
        OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM, &handle_));
    return;
}

template<typename Optix_TransformType>
template<OptixTraversableType TypeEnum>
void MotionTransformBase<Optix_TransformType>::SetUpTraversableHandle_(
    std::span<const std::byte> cpuBuffer)
{
    buffer_ = HostUtils::DeviceMakeUnique<std::byte[]>(cpuBuffer);
    assert(reinterpret_cast<std::uintptr_t>(
               thrust::raw_pointer_cast(buffer_.get())) %
               alignof(Optix_TransformType) ==
           0);
    HostUtils::CheckOptixError(optixConvertPointerToTraversableHandle(
        LocalContextSetter::GetCurrentOptixContext(),
        HostUtils::ToDriverPointer(buffer_.get()), TypeEnum, &handle_));
}

template<typename T>
class MotionTransformBuffer
{
    HostUtils::AlignedBufferType<T> cpuBuffer;
    std::size_t size;

public:
    MotionTransformBuffer(HostUtils::AlignedBufferType<T> &&init_cpuBuffer,
                          std::size_t init_size)
        : cpuBuffer{ std::move(init_cpuBuffer) }, size{ init_size }
    {
    }
    auto GetBufferPtr() const noexcept { return cpuBuffer.get(); }
    auto GetPtr() const noexcept
    {
        return reinterpret_cast<T *>(cpuBuffer.get());
    }
    auto GetSize() const noexcept { return size; }
};

template<typename Optix_TransformType>
static inline MotionTransformBuffer<Optix_TransformType> PrepareMotionTransform(
    std::size_t numKeys, const Traversable &childTraversable,
    decltype(OptixMotionOptions::flags) flags, float tBegin, float tEnd)
{
    HostUtils::CheckError(
        numKeys <=
            std::numeric_limits<decltype(OptixMotionOptions::numKeys)>::max(),
        "Motion keys in matrix transform too large", "Too many motion keys");

    HostUtils::CheckError(numKeys >= 2,
                          "Motion keys in matrix transform too small",
                          "Too few motion keys");

    std::size_t totalSize = sizeof(Optix_TransformType) +
                            (numKeys - 2) * cs_singleMatrixTransformBytesNum;

    auto result =
        HostUtils::MakeAlignedByteBuffer<Optix_TransformType>(totalSize);

    new(result.get()) Optix_TransformType{
        .child = childTraversable.GetHandle(), 
        .motionOptions = {
            .numKeys = static_cast<unsigned short>(numKeys),
            .flags = flags,
            .timeBegin = tBegin,
            .timeEnd = tEnd,
        },
    };
    return { std::move(result), totalSize };
}

static void SetTransformOfMatrixMotionTransform(
    float *dest, std::span<const glm::mat3x4> trans)
{
    assert(trans.size_bytes() ==
           cs_singleMatrixTransformBytesNum * trans.size());

    auto ptr = glm::value_ptr(trans[0]);
    std::copy(ptr, ptr + trans.size(), dest);
    return;
}

static inline MotionTransformBuffer<OptixMatrixMotionTransform>
SetMatrixMotionTransform(std::span<const glm::mat3x4> transforms,
                         const Traversable &childTraversable,
                         decltype(OptixMotionOptions::flags) flags,
                         float tBegin, float tEnd)
{
    auto buffer = PrepareMotionTransform<OptixMatrixMotionTransform>(
        transforms.size(), childTraversable, flags, tBegin, tEnd);
    auto cpuTransformPtr = buffer.GetPtr();
    SetTransformOfMatrixMotionTransform(cpuTransformPtr->transform[0],
                                        transforms);
    return buffer;
}

MatrixMotionTransform::MatrixMotionTransform(
    std::span<const glm::mat3x4> transforms,
    const Traversable &childTraversable, float tBegin, float tEnd)
    : MotionTransformBase{ childTraversable }
{
    auto buffer = SetMatrixMotionTransform(
        transforms, childTraversable, OPTIX_MOTION_FLAG_NONE, tBegin, tEnd);
    SetUpTraversableHandle_<OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM>(
        { buffer.GetBufferPtr(), buffer.GetSize() });
}

MatrixMotionTransform::MatrixMotionTransform(
    std::span<const glm::mat3x4> transforms,
    const Traversable &childTraversable, float tBegin, float tEnd,
    bool beginVanish, bool endVanish)
    : MotionTransformBase{ childTraversable }
{
    auto buffer = SetMatrixMotionTransform(
        transforms, childTraversable,
        (beginVanish ? OPTIX_MOTION_FLAG_START_VANISH : 0) |
            (endVanish ? OPTIX_MOTION_FLAG_END_VANISH : 0),
        tBegin, tEnd);
    SetUpTraversableHandle_<OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM>(
        { buffer.GetBufferPtr(), buffer.GetSize() });
}

static inline MotionTransformBuffer<OptixSRTMotionTransform>
SetSRTMotionTransform(std::span<const OptixSRTData> transforms,
                      const Traversable &childTraversable,
                      decltype(OptixMotionOptions::flags) flags, float tBegin,
                      float tEnd)
{
    auto buffer = PrepareMotionTransform<OptixSRTMotionTransform>(
        transforms.size(), childTraversable, flags, tBegin, tEnd);
    auto cpuSRTDataPtr = buffer.GetPtr();
    std::ranges::copy(transforms, cpuSRTDataPtr->srtData);
    return buffer;
}

SRTMotionTransform::SRTMotionTransform(std::span<const OptixSRTData> transforms,
                                       const Traversable &childTraversable,
                                       float tBegin, float tEnd)
    : MotionTransformBase{ childTraversable }
{
    auto buffer = SetSRTMotionTransform(transforms, childTraversable,
                                        OPTIX_MOTION_FLAG_NONE, tBegin, tEnd);
    SetUpTraversableHandle_<OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM>(
        { buffer.GetBufferPtr(), buffer.GetSize() });
}

SRTMotionTransform::SRTMotionTransform(std::span<const OptixSRTData> transforms,
                                       const Traversable &childTraversable,
                                       float tBegin, float tEnd,
                                       bool beginVanish, bool endVanish)
    : MotionTransformBase{ childTraversable }
{
    auto buffer = SetSRTMotionTransform(
        transforms, childTraversable,
        (beginVanish ? OPTIX_MOTION_FLAG_START_VANISH : 0) |
            (endVanish ? OPTIX_MOTION_FLAG_END_VANISH : 0),
        tBegin, tEnd);
    SetUpTraversableHandle_<OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM>(
        { buffer.GetBufferPtr(), buffer.GetSize() });
}

} // namespace EasyRender::Optix