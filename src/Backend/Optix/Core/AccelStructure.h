#pragma once
#include "BuildInput.h"
#include "HostUtils/DeviceAllocators.h"
#include "HostUtils/EnumUtils.h"
#include "Traversable.h"

namespace Wayland::OptiX
{

/// @brief Build flags is just scoped enumeration for OptixBuildFlags, used by
/// Optix AS.
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
#if OPTIX_VERSION >= 80000
    OpacityMicromapUpdate = OPTIX_BUILD_FLAG_ALLOW_OPACITY_MICROMAP_UPDATE,
    DisableOpacityMicromap = OPTIX_BUILD_FLAG_ALLOW_DISABLE_OPACITY_MICROMAPS
#endif
};

} // namespace Wayland::OptiX
ENABLE_BINARY_OP_FOR_SCOPED_ENUM(Wayland::OptiX::BuildFlags);

namespace Wayland::OptiX
{

/// @brief Abstract class for AS.
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
        Wayland::HostUtils::DeviceUniquePtr<std::byte[]> tempBuffer;
    };

    AccelStructure(const BuildInputArray &arr, BuildFlags flag);
    ASConstructData PrepareBuffer_(const BuildInputArray &arr, BuildFlags flags,
                                   OptixAccelBufferSizes &bufferSizes);

    std::size_t TryCompactAS_(std::size_t outputBufferSize, auto &&accelBuild);

    const BuildInputArray *arrPtr_;
    OptixAccelBuildOptions accelOptions_;
    Wayland::HostUtils::DeviceUniquePtr<std::byte[]> outputBuffer_;
};

/// @brief Non-updatable Optix AS that isn't updatable; this is slightly faster
/// than DynamicAccelStructure, and conveys
/// clear informations to users it have no motion.
/// @sa DynamicAccelStructure
class StaticAccelStructure : public AccelStructure
{
public:
    StaticAccelStructure(const BuildInputArray &arr,
                         BuildFlags flag = BuildFlags::None);
};

/// @brief General Optix AS, allow all BuildFlags.
class DynamicAccelStructure : public AccelStructure
{
public:
    /// @brief Use BuildInput array and BuildFlags (by default None) to
    /// construct any AS.
    /// @param arr Build input array for AS. we don't preserve the pointer to
    /// the build input array in constructor since we hope users to explicitly
    /// manage its lifetime, so that they may not free the array before the last
    /// use of the AS.
    /// @param flag BuildFlags for it.
    DynamicAccelStructure(const BuildInputArray &arr,
                          BuildFlags flag = BuildFlags::None);

    /// @brief Update the AS after updating build inputs.
    /// @param arr same as ctor.
    void Update(const BuildInputArray &arr);

    /// @brief When BVH distortion is significant, \ref{Update} Update will
    /// degenerate the quality of AS. Rebuild is needed to restore its quality.
    /// @param arr same as ctor.
    void Rebuild(const BuildInputArray &arr);

private:
    void EnlargeBuffers_(const OptixBuildInput *buildInputsPtr,
                         BuildInputNumLimitInt buildInputNum);

    std::size_t outputBufferSize_ = 0;

    Wayland::HostUtils::DeviceUniquePtr<std::byte[]> updateBuffer_;
    std::size_t updateBufferSize_ = 0;
};

} // namespace Wayland::OptiX
