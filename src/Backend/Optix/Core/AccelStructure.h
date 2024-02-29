#pragma once
#include "BuildInput.h"
#include "HostUtils/DeviceAllocators.h"
#include "HostUtils/EnumUtils.h"
#include "Traversable.h"

enum class BuildFlags : unsigned int
// decltype(OptixAccelBuildOptions::buildFlags), bug in msvc (solved in 19.39).
{
    None = OPTIX_BUILD_FLAG_NONE,
    Update = OPTIX_BUILD_FLAG_ALLOW_UPDATE,
    Compact = OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
    FastTrace = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
    FastBuild = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD,
    VertexRandomAccess = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS,
    InstanceRandomAccess = OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS,
#if OPTIX_VERSION >= 8000
    OpacityMicromapUpdate = OPTIX_BUILD_FLAG_ALLOW_OPACITY_MICROMAP_UPDATE,
    DisableOpacityMicromap = OPTIX_BUILD_FLAG_ALLOW_DISABLE_OPACITY_MICROMAPS
#endif
};
ENABLE_BINARY_OP_FOR_SCOPED_ENUM(BuildFlags);

class AccelStructure : public Traversable
{
public:
    // Same as parameter type of optix APIs e.g. optixAccelComputeMemoryUsage
    using BuildInputNumLimitInt = unsigned int;

    std::string DisplayInfo() const override
    {
        return "Acceleration structure.";
    }

    unsigned int GetDepth() const noexcept override
    {
        return arrPtr_->GetDepth();
    }

protected:
    struct ASConstructData
    {
        const OptixBuildInput *buildInputsPtr;
        BuildInputNumLimitInt buildInputNum;
        HostUtils::DeviceUniquePtr<std::byte[]> tempBuffer;
    };

    AccelStructure(const BuildInputArray &arr, BuildFlags flag);
    ASConstructData PrepareBuffer_(const BuildInputArray &arr, BuildFlags flags,
                                   OptixAccelBufferSizes &bufferSizes);

    std::size_t TryCompactAS_(std::size_t outputBufferSize, auto &&accelBuild);

    const BuildInputArray *arrPtr_;
    OptixAccelBuildOptions accelOptions_;
    HostUtils::DeviceUniquePtr<std::byte[]> outputBuffer_;
};

class StaticAccelStructure : public AccelStructure
{
public:
    StaticAccelStructure(const BuildInputArray &arr,
                         BuildFlags flag = BuildFlags::None);
};

class DynamicAccelStructure : public AccelStructure
{
public:
    DynamicAccelStructure(const BuildInputArray &arr,
                          BuildFlags flag = BuildFlags::None);

    /// @brief Update the AS after updating build inputs.
    /// @param arr Same as constructor; we don't preserve the pointer to the
    /// build input array in constructor since we hope users to explicitly
    /// manage its lifetime, so that they may not free the array before the last
    /// use of the AS.
    void Update(const BuildInputArray &arr);

    void Rebuild(const BuildInputArray &arr);

private:
    void EnlargeBuffers_(const OptixBuildInput *buildInputsPtr,
                         BuildInputNumLimitInt buildInputNum);

    std::size_t outputBufferSize_ = 0;

    HostUtils::DeviceUniquePtr<std::byte[]> updateBuffer_;
    std::size_t updateBufferSize_ = 0;
};
