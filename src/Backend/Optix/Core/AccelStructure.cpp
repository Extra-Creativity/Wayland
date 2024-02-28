#include "AccelStructure.h"
#include "ContextManager.h"

#include <limits>
#include <spdlog/spdlog.h>

#undef max
#undef min

static auto CheckBuildInputArrLimit(const auto &buildInputs)
{
    using LimitInt = AccelStructure::BuildInputNumLimitInt;

    auto buildInputNum = buildInputs.size();
    HostUtils::CheckError(buildInputNum >= 1, "No build inputs.",
                          "At least one build input should be provided to "
                          "build acceleration structure.");
    HostUtils::CheckError(
        buildInputNum <= std::numeric_limits<LimitInt>::max(),
        "Too many build inputs.",
        "Number of build inputs should be within limits of unsigned int");
    return static_cast<LimitInt>(buildInputNum);
}

AccelStructure::AccelStructure(const BuildInputArray &arr, BuildFlags flags)
    : arrPtr_{ &arr }, accelOptions_{ .buildFlags = std::to_underlying(flags),
                                      .operation = OPTIX_BUILD_OPERATION_BUILD,
                                      .motionOptions = { .numKeys = 1 } }
{
}

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

std::size_t AccelStructure::TryCompactAS_(std::size_t outputBufferSize,
                                          auto &&accelBuild)
{
    std::size_t compactSize;
    OptixAccelEmitDesc desc{
        .type = OptixAccelPropertyType::OPTIX_PROPERTY_TYPE_COMPACTED_SIZE
    };
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
    }
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
        newSize < updateBufferSize_)
    {
        updateBuffer_.reset();
        updateBuffer_ =
            HostUtils::DeviceMakeUninitializedUnique<std::byte[]>(newSize);
        updateBufferSize_ = newSize;
    }
}