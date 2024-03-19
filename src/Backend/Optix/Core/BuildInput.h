#pragma once
#include "HostUtils/CommonHeaders.h"
#include "HostUtils/DeviceAllocators.h"
#include "HostUtils/EnumUtils.h"
#include <span>

namespace Wayland::OptiX
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

// @brief Store either a flag or an array of flags; when flag number is one, use
// the single flag; otherwise allocate dynamically. Used by primitive types like
// triangles.
// @note Why not std::variant: we want to use "index" of variant to store the
// number (which is also unique!), to reduce the space needed.
class FlagVariant
{
    using UnderlyingType = std::underlying_type_t<GeometryFlags>;

    std::size_t flagNum_;
    union Data {
        Data() {}
        UnderlyingType singleFlag;
        std::unique_ptr<UnderlyingType[]> flags;
        ~Data() {}
    } data_;
    bool IsSingleFlag_() const noexcept { return flagNum_ == 1; }
    void Clear_() const noexcept
    {
        if (!IsSingleFlag_())
            data_.flags.~unique_ptr();
    }

public:
    auto GetFlagNum() const noexcept { return flagNum_; }
    auto GetFlagPtr() const noexcept
    {
        return IsSingleFlag_() ? &data_.singleFlag : data_.flags.get();
    }

    FlagVariant(GeometryFlags flag) : flagNum_{ 1 }
    {
        data_.singleFlag = std::to_underlying(flag);
    }
    ~FlagVariant() { Clear_(); }
    FlagVariant(std::span<const GeometryFlags> flags);
    FlagVariant(FlagVariant &&another) noexcept;
    FlagVariant &operator=(FlagVariant &&another) noexcept;
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
    auto GetFlagNum() const noexcept { return flagBuffer_.GetFlagNum(); }
    auto GetFlagPtr() const noexcept { return flagBuffer_.GetFlagPtr(); }
    auto GetSBTIndexOffsetBuffer() const noexcept
    {
        return Wayland::HostUtils::ToDriverPointer(sbtIndexOffsetBuffer_.get());
    }

private:
    FlagVariant flagBuffer_;
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

/// @brief Buildinput array for triangle type, mostly used type.
class TriangleBuildInputArray : public BuildInputArray
{
    void GeneralAddBuildInput_(auto &&);

public:
    TriangleBuildInputArray() = default;
    /// @brief Create build input array, with reserved memory to prevent
    /// reallocation when AddBuildInput
    /// @param expectBuildInputNum expected number of build inputs so that
    /// memory will be reserved to proper size.
    TriangleBuildInputArray(std::size_t expectBuildInputNum)
        : BuildInputArray{ expectBuildInputNum }
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

private:
    std::vector<TriangleDataBuffer> dataBuffers_;
};

/// @brief data buffer of instance build input array; this shouldn't be used
/// by others, but InstanceBuildInputArray.
class InstanceDataBuffer
{
public:
    InstanceDataBuffer(auto instances);
    auto GetInstanceBufferPtr() const noexcept
    {
        return Wayland::HostUtils::ToDriverPointer(instancesBuffer_.get());
    }

private:
    Wayland::HostUtils::DeviceUniquePtr<OptixInstance[]> instancesBuffer_;
};

class Traversable;
class InstanceBuildInputArray : public BuildInputArray
{
public:
    InstanceBuildInputArray() = default;
    /// @brief Create build input array, with reserved memory to prevent
    /// reallocation when AddBuildInput.
    /// @param expectBuildInputNum expected number of build inputs so that
    /// memory will be reserved to proper size.
    InstanceBuildInputArray(std::size_t expectBuildInputNum)
        : BuildInputArray{ expectBuildInputNum }
    {
    }

    /// @brief Add a build input constructed by parameters.
    /// @param instances instances to be added, whose traversable handle can be
    /// unset (i.e. the ctor will set it).
    /// @param children children of the instances, used to get depth and set
    /// traverable handles.
    void AddBuildInput(std::span<OptixInstance> instances,
                       std::span<const Traversable *> children);

    void RemoveBuildInput(std::size_t idx) noexcept;

    unsigned int GetDepth() const noexcept override;

private:
    /// @brief This member is used to get depth of the traversable, otherwise
    /// unnecessary.
    std::vector<std::vector<const Traversable *>> children_;
    std::vector<InstanceDataBuffer> dataBuffers_;
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
} // namespace Wayland::OptiX

ENABLE_BINARY_OP_FOR_SCOPED_ENUM(Wayland::OptiX::GeometryFlags);
ENABLE_BINARY_OP_FOR_SCOPED_ENUM(Wayland::OptiX::InstanceFlags);