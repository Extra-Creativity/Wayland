#include "AccelStructure.h"
#include "ContextManager.h"

#include <limits>
#include <spdlog/spdlog.h>

#undef max
#undef min

using namespace EasyRender;

static auto CheckBuildInputArrLimit(const auto &buildInputs)
{
    using LimitInt = Optix::AccelStructure::BuildInputNumLimitInt;

    auto buildInputNum = buildInputs.size();
    HostUtils::CheckError(buildInputNum >= 1, "No build inputs.",
                          "At least one build input should be provided to "
                          "build acceleration structure.");
    HostUtils::CheckError(
        HostUtils::CheckInRange<LimitInt>(buildInputNum),
        "Too many build inputs.",
        "Number of build inputs should be within limits of unsigned int");
    return static_cast<LimitInt>(buildInputNum);
}

namespace EasyRender::Optix
{

AccelStructure::AccelStructure(const BuildInputArray &arr, BuildFlags flags)
    : arrPtr_{ &arr }, accelOptions_{ .buildFlags = std::to_underlying(flags),
                                      .operation = OPTIX_BUILD_OPERATION_BUILD,
                                      .motionOptions = { .numKeys = 1 } }
{
}

/// @brief Prepare the output buffer and return necessary information for
/// optixBuild. The temporary buffer is returned by return value because
/// StaticAS doesn't need to preserve it after build, so it's the duty of caller
/// to decide how to process the temporary buffer.
/// @param arr build input array of the current build.
/// @param flags build flag of the current build.
/// @param[out] bufferSizes fill buffer sizes with proper values.
/// @return ASConstructData, i.e. A) buildInputsPtr, raw pointer of arr.
/// B) buildInputNum, checked size with proper integer type.
/// C) tempBuffer, allocated temporary buffer.
AccelStructure::ASConstructData AccelStructure::PrepareBuffer_(
    const BuildInputArray &arr, BuildFlags flags,
    OptixAccelBufferSizes &bufferSizes)
{
    const auto &buildInputs = arr.GetBuildInputArr();
    auto buildInputNum = CheckBuildInputArrLimit(buildInputs);

    HostUtils::CheckOptixError(optixAccelComputeMemoryUsage(
        LocalContextSetter::GetCurrentOptixContext(), &accelOptions_,
        buildInputs.data(), buildInputNum, &bufferSizes));

    outputBuffer_ = HostUtils::DeviceMakeUninitializedUnique<std::byte[]>(
        bufferSizes.outputSizeInBytes);
    return { buildInputs.data(), buildInputNum,
             HostUtils::DeviceMakeUninitializedUnique<std::byte[]>(
                 bufferSizes.tempSizeInBytes) };
}

/// @brief Compact AS if
/// @param outputBufferSize original size of the output buffer.
/// @param accelBuild handle to build the acceleration, which should accept
/// (const OptixAccelEmitDesc *emittedProperties, unsigned int propertyNum) to
/// call optixBuild.
/// @return new size of the output buffer, possibly compacted.
std::size_t AccelStructure::TryCompactAS_(std::size_t outputBufferSize,
                                          auto &&accelBuild)
{
    std::size_t compactSize;
    OptixAccelEmitDesc desc{
        .type = OptixAccelPropertyType::OPTIX_PROPERTY_TYPE_COMPACTED_SIZE
    };
    // get compacted size; use unified memory if possible.
    if (LocalContextSetter::CurrentCanAsyncAccessUnifiedMemory())
    {
        auto compactSizePtr = HostUtils::DeviceMakeUninitializedUnique<
            std::uint64_t, HostUtils::DeviceManagedAllocator<std::uint64_t>>();
        desc.result = HostUtils::ToDriverPointer(compactSizePtr.get()),
        accelBuild(&desc, 1);
        compactSize = *compactSizePtr;
    }
    else // e.g. for windows or wsl.
    {
        auto compactSizePtr =
            HostUtils::DeviceMakeUninitializedUnique<std::uint64_t>();
        desc.result = HostUtils::ToDriverPointer(compactSizePtr.get()),
        accelBuild(&desc, 1);
        thrust::copy_n(compactSizePtr.get(), 1, &compactSize);
    }

    if (compactSize < outputBufferSize) [[likely]]
    {
        auto compactBuffer =
            HostUtils::DeviceMakeUninitializedUnique<std::byte[]>(compactSize);
        HostUtils::CheckOptixError(optixAccelCompact(
            LocalContextSetter::GetCurrentOptixContext(),
            LocalContextSetter::GetCurrentCUDAStream(), handle_,
            HostUtils::ToDriverPointer(compactBuffer.get()), compactSize,
            &handle_));
        outputBuffer_ = std::move(compactBuffer);
        return compactSize;
    }
    SPDLOG_WARN("Unable to compact the AS.");
    return outputBufferSize;
}

StaticAccelStructure::StaticAccelStructure(const BuildInputArray &arr,
                                           BuildFlags flags)
    : AccelStructure{ arr, flags }
{
    HostUtils::CheckError(!HostUtils::TestEnum(flags, BuildFlags::Update),
                          "Static AS shouldn't contain Update flag.");

    OptixAccelBufferSizes bufferSizes;
    auto data = PrepareBuffer_(arr, flags, bufferSizes);

    auto accelBuild = [&, this](const OptixAccelEmitDesc *emittedProperties,
                                unsigned int numEmittedProperties) {
        HostUtils::CheckOptixError(optixAccelBuild(
            LocalContextSetter::GetCurrentOptixContext(),
            LocalContextSetter::GetCurrentCUDAStream(), &accelOptions_,
            data.buildInputsPtr, data.buildInputNum,
            HostUtils::ToDriverPointer(data.tempBuffer.get()),
            bufferSizes.tempSizeInBytes,
            HostUtils::ToDriverPointer(outputBuffer_.get()),
            bufferSizes.outputSizeInBytes, &handle_, emittedProperties,
            numEmittedProperties));
    };

    if (!HostUtils::TestEnum(flags, BuildFlags::Compact))
    {
        accelBuild(nullptr, 0);
        return;
    } // build directly, else try to compact.
    TryCompactAS_(bufferSizes.outputSizeInBytes, accelBuild);
}

DynamicAccelStructure::DynamicAccelStructure(const BuildInputArray &arr,
                                             BuildFlags flags)
    : AccelStructure{ arr, flags }
{
    OptixAccelBufferSizes bufferSizes;
    auto data = PrepareBuffer_(arr, flags, bufferSizes);

    auto accelBuild = [&, this](const OptixAccelEmitDesc *emittedProperties,
                                unsigned int numEmittedProperties) {
        auto &tempBuffer = data.tempBuffer;
        HostUtils::CheckOptixError(optixAccelBuild(
            LocalContextSetter::GetCurrentOptixContext(),
            LocalContextSetter::GetCurrentCUDAStream(), &accelOptions_,
            data.buildInputsPtr, data.buildInputNum,
            HostUtils::ToDriverPointer(tempBuffer.get()),
            bufferSizes.tempSizeInBytes,
            HostUtils::ToDriverPointer(outputBuffer_.get()),
            bufferSizes.outputSizeInBytes, &handle_, emittedProperties,
            numEmittedProperties));
        outputBufferSize_ = bufferSizes.outputSizeInBytes;
        if (HostUtils::TestEnum(flags, BuildFlags::Update))
        {
            // if temp buffer is big enough, use it as update buffer directly.
            if (bufferSizes.tempUpdateSizeInBytes <=
                bufferSizes.tempSizeInBytes)
            {
                updateBuffer_ = std::move(tempBuffer);
                updateBufferSize_ = bufferSizes.tempSizeInBytes;
            }
            else
            {
                tempBuffer.reset(); // release useless memory first.
                updateBuffer_ =
                    HostUtils::DeviceMakeUninitializedUnique<std::byte[]>(
                        bufferSizes.tempUpdateSizeInBytes);
                updateBufferSize_ = bufferSizes.tempUpdateSizeInBytes;
            }
        }
    };

    if (!HostUtils::TestEnum(flags, BuildFlags::Compact))
    {
        accelBuild(nullptr, 0);
        return;
    }
    // build directly, else try to compact.
    outputBufferSize_ =
        TryCompactAS_(bufferSizes.outputSizeInBytes, accelBuild);
}

void DynamicAccelStructure::Update(const BuildInputArray &arr)
{
    HostUtils::CheckError(
        HostUtils::TestEnum(accelOptions_.buildFlags, BuildFlags::Update),
        "Acceleration structure isn't updatable.",
        "Invalid update of acceleration structure");
    assert(updateBuffer_ && updateBufferSize_); // ensure not empty buffer

    const auto &buildInputs = arr.GetBuildInputArr();
    auto buildInputNum = CheckBuildInputArrLimit(buildInputs);

    accelOptions_.operation = OPTIX_BUILD_OPERATION_UPDATE;

    HostUtils::CheckOptixError(optixAccelBuild(
        LocalContextSetter::GetCurrentOptixContext(),
        LocalContextSetter::GetCurrentCUDAStream(), &accelOptions_,
        buildInputs.data(), buildInputNum,
        HostUtils::ToDriverPointer(updateBuffer_.get()), updateBufferSize_,
        HostUtils::ToDriverPointer(outputBuffer_.get()), outputBufferSize_,
        &handle_, nullptr, 0));
}

void DynamicAccelStructure::Rebuild(const BuildInputArray &arr)
{
    HostUtils::CheckError(
        HostUtils::TestEnum(accelOptions_.buildFlags, BuildFlags::Update),
        "Non-updateable acceleration structure doesn't need to rebuild",
        "Redudant build of acceleration structure");

    const auto &buildInputs = arr.GetBuildInputArr();
    auto buildInputNum = CheckBuildInputArrLimit(buildInputs);

    accelOptions_.operation = OPTIX_BUILD_OPERATION_BUILD;

    auto accelBuild = [&, this](const OptixAccelEmitDesc *emittedProperties,
                                unsigned int numEmittedProperties) {
        HostUtils::CheckOptixError(optixAccelBuild(
            LocalContextSetter::GetCurrentOptixContext(),
            LocalContextSetter::GetCurrentCUDAStream(), &accelOptions_,
            buildInputs.data(), buildInputNum,
            HostUtils::ToDriverPointer(updateBuffer_.get()), updateBufferSize_,
            HostUtils::ToDriverPointer(outputBuffer_.get()), outputBufferSize_,
            &handle_, emittedProperties, numEmittedProperties));
    };

    if (!HostUtils::TestEnum(accelOptions_.buildFlags, BuildFlags::Compact))
    {
        accelBuild(nullptr, 0);
        return;
    }

    // Compacted structure may need to change the size of buffers.
    EnlargeBuffers_(buildInputs.data(), buildInputNum);
    outputBufferSize_ = TryCompactAS_(outputBufferSize_, accelBuild);
}

/// @brief Enlarge the output buffer to make it suitable for rebuild.
/// @param buildInputsPtr pointer of buildinputs
/// @param buildInputNum checked size of build input.
void DynamicAccelStructure::EnlargeBuffers_(
    const OptixBuildInput *buildInputsPtr, BuildInputNumLimitInt buildInputNum)
{
    OptixAccelBufferSizes bufferSizes;
    HostUtils::CheckOptixError(optixAccelComputeMemoryUsage(
        LocalContextSetter::GetCurrentOptixContext(), &accelOptions_,
        buildInputsPtr, buildInputNum, &bufferSizes));
    if (bufferSizes.outputSizeInBytes > outputBufferSize_)
    {
        outputBuffer_.reset(); // release memory first.
        outputBuffer_ = HostUtils::DeviceMakeUninitializedUnique<std::byte[]>(
            bufferSizes.outputSizeInBytes);
        outputBufferSize_ = bufferSizes.outputSizeInBytes;
    }
    if (auto newSize = std::max(bufferSizes.tempSizeInBytes,
                                bufferSizes.tempUpdateSizeInBytes);
        newSize > updateBufferSize_)
    {
        updateBuffer_.reset();
        updateBuffer_ =
            HostUtils::DeviceMakeUninitializedUnique<std::byte[]>(newSize);
        updateBufferSize_ = newSize;
    }
}

} // namespace EasyRender::Optix