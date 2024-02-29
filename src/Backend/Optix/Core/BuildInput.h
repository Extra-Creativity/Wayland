#pragma once
#include "HostUtils/CommonHeaders.h"
#include "HostUtils/DeviceAllocators.h"
#include "HostUtils/EnumUtils.h"
#include <span>

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

// Used by triangle and customized primitives.
enum class GeometryFlags : unsigned int
// i.e. decltype(*OptixBuildInputTriangleArray::flags)
{
    None = OPTIX_GEOMETRY_FLAG_NONE,
    DiableAnyHit = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    SingleAnyhit = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL,
    FaceCulling = OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING
};
ENABLE_BINARY_OP_FOR_SCOPED_ENUM(GeometryFlags);

// Why not std::variant: we want to use "index" of variant to store the
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

class TriangleDataBuffer
{
public:
    TriangleDataBuffer(const auto &vertices, auto triangles, auto flags);
    TriangleDataBuffer(const auto &vertices, auto triangles, auto flags,
                       auto sbtOffsets);

    auto GetVertexPtrsPtr() const noexcept { return vertexPtrArr_.get(); }
    auto GetTrianglesPtr() const noexcept
    {
        return HostUtils::ToDriverPointer(trianglesBuffer_.get());
    }
    auto GetMotionKeyNum() const noexcept { return motionKeyNum_; }
    auto GetFlagNum() const noexcept { return flagBuffer_.GetFlagNum(); }
    auto GetFlagPtr() const noexcept { return flagBuffer_.GetFlagPtr(); }
    auto GetSBTIndexOffsetBuffer() const noexcept
    {
        return HostUtils::ToDriverPointer(sbtIndexOffsetBuffer_.get());
    }

private:
    FlagVariant flagBuffer_;
    HostUtils::DeviceUniquePtr<std::byte[]> sbtIndexOffsetBuffer_;
    HostUtils::DeviceUniquePtr<std::byte[]> verticesBuffer_;
    HostUtils::DeviceUniquePtr<std::byte[]> trianglesBuffer_;
    std::unique_ptr<CUdeviceptr[]> vertexPtrArr_;
    /// @brief We don't restrict it to limits of optix motion key type
    /// until it's linked against the parent?
    std::size_t motionKeyNum_;
};

class TriangleBuildInputArray : public BuildInputArray
{
    void GeneralAddBuildInput_(auto &&);

public:
    TriangleBuildInputArray() = default;
    TriangleBuildInputArray(std::size_t expectBuildInputNum)
        : BuildInputArray{ expectBuildInputNum }
    {
        dataBuffers_.reserve(expectBuildInputNum);
    }

    // TODO: Add more APIs, including short and char (for random stride).
    void AddBuildInput(const std::vector<std::span<const float>> &vertices,
                       std::span<const int> triangles,
                       GeometryFlags flag = GeometryFlags::None);
    void AddBuildInput(const std::vector<std::span<const float>> &vertices,
                       std::span<const int> triangles,
                       std::span<GeometryFlags> flags,
                       std::span<const std::uint32_t> sbtIndexOffset);
    void RemoveBuildInput(std::size_t idx) noexcept;

    unsigned int GetDepth() const noexcept override { return 1; }

private:
    std::vector<TriangleDataBuffer> dataBuffers_;
};

class InstanceDataBuffer
{
public:
    InstanceDataBuffer(auto instances);
    auto GetInstanceBufferPtr() const noexcept
    {
        return HostUtils::ToDriverPointer(instancesBuffer_.get());
    }

private:
    HostUtils::DeviceUniquePtr<OptixInstance[]> instancesBuffer_;
};

class Traversable;
class InstanceBuildInputArray : public BuildInputArray
{
public:
    InstanceBuildInputArray() = default;
    InstanceBuildInputArray(std::size_t expectBuildInputNum)
        : BuildInputArray{ expectBuildInputNum }
    {
    }

    void AddBuildInput(std::span<const OptixInstance> instances,
                       std::span<const Traversable *> children);

    void RemoveBuildInput(std::size_t idx) noexcept;

    unsigned int GetDepth() const noexcept override;

private:
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
#if OPTIX_VERSION >= 8000
    ForceOMM2State = OPTIX_INSTANCE_FLAG_FORCE_OPACITY_MICROMAP_2_STATE,
    DisableOMM = OPTIX_INSTANCE_FLAG_DISABLE_OPACITY_MICROMAPS
#endif
};
ENABLE_BINARY_OP_FOR_SCOPED_ENUM(InstanceFlags);