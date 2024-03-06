#pragma once
#include "HostUtils/CommonHeaders.h"
#include "HostUtils/CompactVariant.h"
#include "HostUtils/DeviceAllocators.h"
#include "HostUtils/EnumUtils.h"

#include "SBTData.h"

#include <any>
#include <functional>
#include <span>
#include <vector>

#include "thrust/device_vector.h"

namespace Wayland::Optix
{

/// @brief Base class of all build input arrays, like
/// \ref{TriangleBuildInputArray} TriangleBuildInputArray. The reason why we
/// group them together is that sizeof(OptixBuildInput) is really huge, so we
/// hope not to gather them when AS is built, but just make them occupy
/// contiguous memory directly. This may usually cooperate with data buffer.
class BuildInputArray
{
public:
    BuildInputArray() = default;
    BuildInputArray(std::size_t expectBuildInputNum)
    {
        buildInputs_.reserve(expectBuildInputNum);
    }

    const auto &GetBuildInputArr() const noexcept { return buildInputs_; }

    void RemoveBuildInput(std::size_t idx) noexcept
    {
        buildInputs_.erase(buildInputs_.begin() + idx);
    }

    virtual unsigned int GetDepth() const noexcept = 0;

protected:
    std::vector<OptixBuildInput> buildInputs_;
};

// Used by primitives types.
enum class GeometryFlags : unsigned int
// i.e. decltype(*OptixBuildInputTriangleArray::flags)
{
    None = OPTIX_GEOMETRY_FLAG_NONE,
    DiableAnyHit = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    SingleAnyhit = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL,
    FaceCulling = OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING
};

/// @brief data buffer of triangle build input array; this shouldn't be used
/// by others, but TriangleBuildInputArray.
class TriangleDataBuffer
{
public:
    TriangleDataBuffer(const auto &vertices, auto triangles, auto flags);
    TriangleDataBuffer(const auto &vertices, auto triangles, auto flags,
                       auto sbtOffsets);

    auto GetVertexPtrsPtr() const noexcept { return vertexPtrArr_.get(); }
    auto GetTrianglesPtr() const noexcept
    {
        return Wayland::HostUtils::ToDriverPointer(trianglesBuffer_.get());
    }
    auto GetMotionKeyNum() const noexcept { return motionKeyNum_; }
    auto GetFlagNum() const noexcept { return flagBuffer_.GetSize(); }
    auto GetFlagPtr() const noexcept { return flagBuffer_.GetPtr(); }
    auto GetSBTIndexOffsetBuffer() const noexcept
    {
        return Wayland::HostUtils::ToDriverPointer(sbtIndexOffsetBuffer_.get());
    }

private:
    Wayland::HostUtils::CompactVariant<GeometryFlags> flagBuffer_;
    Wayland::HostUtils::DeviceUniquePtr<std::byte[]> sbtIndexOffsetBuffer_;
    Wayland::HostUtils::DeviceUniquePtr<std::byte[]> verticesBuffer_;
    Wayland::HostUtils::DeviceUniquePtr<std::byte[]> trianglesBuffer_;
    // Those with motion may need more than one vertex pointer. This may
    // also be optimized like FlagVariant to reduce one heap allocation.
    std::unique_ptr<CUdeviceptr[]> vertexPtrArr_;
    // We don't restrict it to limits of optix motion key type
    // until it's linked against the parent.
    std::size_t motionKeyNum_;
};

/// @details Parameters' meaning:
/// + unsigned int: build input id.
/// + unsigned int: sbt record id (<= numSbtRecord of the build input);
/// + unsigned int: ray type.
template<typename T>
using SBTSetter =
    std::function<SBTData<T>(unsigned int, unsigned int, unsigned int)>;

class GeometryBuildInputArray : public BuildInputArray
{
public:
    GeometryBuildInputArray() = default;
    GeometryBuildInputArray(std::size_t expectBuildInputNum)
        : BuildInputArray{ expectBuildInputNum }
    {
    }

    using SBTSetterProxy =
        std::function<void(unsigned int, unsigned int, unsigned int, std::any)>;

    /// @note Set an automatic setter for build inputs.
    template<typename T>
    void SetSBTSetter(SBTSetter<T> &&init_setter)
    {
        sbtSetter_ = [setter =
                          std::forward<decltype(init_setter)>(init_setter)](
                         unsigned int buildInputID, unsigned int sbtRecordID,
                         unsigned int rayType, std::any &sbtBuffer) {
            using ContainerType = std::invoke_result_t<
                decltype(GetSBTHitRecordBuffer<SBTData<T>>), unsigned int,
                const Traversable &>;
            auto buffer = std::any_cast<ContainerType>(&sbtBuffer);
            HostUtils::CheckError(buffer,
                                  "Type of buffer and setter doesn't match.");
            buffer->emplace_back(setter(buildInputID, sbtRecordID, rayType));
        };
    }

    const auto &GetSBTSetterProxy() const & { return sbtSetter_; }
    auto GetSBTSetterProxy() && { return std::move(sbtSetter_); }

    struct SBTSetterParamInfo
    {
        unsigned int buildInputNum_;
        std::unique_ptr<unsigned int[]> sbtRecordIDs_;
    };

    virtual SBTSetterParamInfo GetSBTSetterParamInfo() const = 0;

private:
    SBTSetterProxy sbtSetter_;
};

/// @brief Buildinput array for triangle type, mostly used type.
class TriangleBuildInputArray : public GeometryBuildInputArray
{
    void GeneralAddBuildInput_(auto &&);

public:
    TriangleBuildInputArray() = default;
    /// @brief Create build input array, with reserved memory to prevent
    /// reallocation when AddBuildInput
    /// @param expectBuildInputNum expected number of build inputs so that
    /// memory will be reserved to proper size.
    TriangleBuildInputArray(std::size_t expectBuildInputNum)
        : GeometryBuildInputArray{ expectBuildInputNum }
    {
        dataBuffers_.reserve(expectBuildInputNum);
    }

    // TODO: Add more APIs, including short and char (for random stride).

    /// @brief Add a build input constructed by parameters.
    /// @param vertices vector of vertices; the length of vector represents
    /// number of motion key, with each element vertex sequence of that.
    /// @param triangles triangle sequence, it's unrelated to motion.
    /// @param flag a single geometry flag, usual case for a whole mesh.
    void AddBuildInput(const std::vector<std::span<const float>> &vertices,
                       std::span<const int> triangles,
                       GeometryFlags flag = GeometryFlags::None);

    /// @param flags use more than one flag, whose size specifies numSbtRecords.
    /// @param sbtIndexOffset use sbt index to set every primitive, should have
    /// the same size as triangles.
    void AddBuildInput(const std::vector<std::span<const float>> &vertices,
                       std::span<const int> triangles,
                       std::span<GeometryFlags> flags,
                       std::span<const std::uint32_t> sbtIndexOffset);
    void RemoveBuildInput(std::size_t idx) noexcept;

    unsigned int GetDepth() const noexcept override { return 1; }

    SBTSetterParamInfo GetSBTSetterParamInfo() const override;

private:
    std::vector<TriangleDataBuffer> dataBuffers_;
};

class Traversable;
class InstanceBuildInputArray : public BuildInputArray
{
public:
    InstanceBuildInputArray() { buildInputs_.resize(1); }
    /// @brief Create build input array, with reserved memory to prevent
    /// reallocation when AddBuildInput.
    /// @param expectBuildInputNum expected number of build inputs so that
    /// memory will be reserved to proper size.
    InstanceBuildInputArray(std::size_t expectBuildInputNum)
    {
        buildInputs_.resize(1);
        deviceInstances_.reserve(expectBuildInputNum);
    }

    /// @brief Add a build input constructed by parameters.
    /// @param instances instances to be added, whose traversable handle can be
    /// not set (i.e. the ctor will set it).
    /// @param child child of the instance, used to get depth and set
    /// traverable handles.
    void AddBuildInput(OptixInstance &instance, const Traversable *child);

    void RemoveBuildInput(std::size_t idx) noexcept;

    unsigned int GetDepth() const noexcept override;

    const auto &GetChildren() const noexcept { return children_; }

    const auto &GetInstances() const noexcept { return instances_; }

    void SyncToDevice()
    {
        auto size = instances_.size();
        if (deviceInstances_.size() < size)
            deviceInstances_.resize(size);

        thrust::copy(instances_.begin(), instances_.end(),
                     deviceInstances_.begin());
        assert(!buildInputs_.empty());
        auto &instanceArr = buildInputs_.front().instanceArray;
        instanceArr.instances =
            HostUtils::ToDriverPointer(deviceInstances_.data());
        instanceArr.numInstances = size;
    }

private:
    /// @brief This member is used to get depth of the traversable, otherwise
    /// unnecessary.
    std::vector<const Traversable *> children_;
    std::vector<OptixInstance> instances_;
    thrust::device_vector<OptixInstance> deviceInstances_;
};

enum class InstanceFlags : unsigned int
// i.e. decltype(OptixInstance::flags)
{
    None = OPTIX_INSTANCE_FLAG_NONE,
    DisableTriangleFaceCulling =
        OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING,
    FlipTriangleFacing = OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING,
    DisableAnyHit = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT,
    EnforceAnyHit = OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT,
#if OPTIX_VERSION >= 80000
    ForceOMM2State = OPTIX_INSTANCE_FLAG_FORCE_OPACITY_MICROMAP_2_STATE,
    DisableOMM = OPTIX_INSTANCE_FLAG_DISABLE_OPACITY_MICROMAPS
#endif
};
} // namespace Wayland::Optix

ENABLE_BINARY_OP_FOR_SCOPED_ENUM(Wayland::Optix::GeometryFlags);
ENABLE_BINARY_OP_FOR_SCOPED_ENUM(Wayland::Optix::InstanceFlags);