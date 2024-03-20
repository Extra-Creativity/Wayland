#include "BuildInput.h"
#include "Traversable.h"

#include "HostUtils/DebugUtils.h"
#include "HostUtils/ErrorCheck.h"
#include "UniUtils/MathUtils.h"

#include <algorithm>
#include <cassert>
#include <ranges>
#include <type_traits>

#undef max
#undef min

using namespace Wayland;

namespace Wayland::Optix
{

FlagVariant::FlagVariant(std::span<const GeometryFlags> flags)
{
    if (auto size = flags.size(); size == 0)
        flagNum_ = 1, data_.singleFlag = OPTIX_GEOMETRY_FLAG_NONE;
    else if (size == 1)
        flagNum_ = 1, data_.singleFlag = (UnderlyingType)flags[0];
    else [[likely]]
    {
        flagNum_ = size;
        new (&data_) decltype(data_.flags){
            std::make_unique_for_overwrite<UnderlyingType[]>(size)
        };
        std::ranges::copy_n(
            reinterpret_cast<const UnderlyingType *>(flags.data()), size,
            data_.flags.get());
    }
}

FlagVariant::FlagVariant(FlagVariant &&another) noexcept
    : flagNum_{ another.flagNum_ }
{
    if (IsSingleFlag_())
        data_.singleFlag = another.data_.singleFlag;
    else
        new (&data_) decltype(data_.flags){ std::move(another.data_.flags) };
}

FlagVariant &FlagVariant::operator=(FlagVariant &&another) noexcept
{
    Clear_();
    flagNum_ = another.flagNum_;
    if (IsSingleFlag_())
        data_.singleFlag = another.data_.singleFlag;
    else
        new (&data_) decltype(data_.flags){ std::move(another.data_.flags) };

    return *this;
}

using TriangleVertexNumType =
    decltype(OptixBuildInputTriangleArray::numVertices);
using TriangleIndexNumType =
    decltype(OptixBuildInputTriangleArray::numIndexTriplets);

/// @brief Check all preconditions for vertex array and triangle, like having
/// proper size, the triangle index is within the number of vertex, etc., and
/// set type of build input as triangles.
/// @param[out] buildInput build input whose type will be filled.
/// @param verticesArr array of vertex span
/// @param triangles span of triangle index; copied by value, so not expect a
/// owning type like vector.
/// @return pair of vertex number and triangle number, both are with proper type
/// and within proper limit.
static inline std::pair<TriangleVertexNumType, TriangleIndexNumType>
PrepareTriangleBuildInput(OptixBuildInput &buildInput, const auto &verticesArr,
                          auto triangles)
{
    TriangleIndexNumType triangleNum;
    HostUtils::CheckError(
        HostUtils::CheckInRangeAndSet(std::size(triangles), triangleNum),
        "Too many triangles");
    assert(triangleNum > 0 && triangleNum % 3 == 0 && !std::empty(verticesArr));
    triangleNum /= 3;

    TriangleVertexNumType vertNum;
    HostUtils::CheckError(
        HostUtils::CheckInRangeAndSet(std::size(verticesArr[0]), vertNum),
        "Too many vertices");
    assert(vertNum > 0 && vertNum % 3 == 0);
    vertNum = vertNum / 3;

#ifdef BUILD_INPUT_CHECK_SAME_VERTICES_NUM
    HostUtils::CheckError(
        std::ranges::all_of(
            vertices,
            [vertNum](auto &vertices) { return vertices.size() == vertNum; }),
        "Vertices should be of the same amount in all frames.");
#endif

#ifdef BUILD_INPUT_CHECK_TRIANGLE_IN_RANGE
    HostUtils::CheckError(
        std::ranges::all_of(triangles,
                            [vertNum](auto idx) { return idx < vertNum; }),
        "Triangle index is greater than vertice number");
#endif
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    return { vertNum, triangleNum };
}

static inline void FillTriangleBuildInputFormatAndStrides(
    OptixBuildInput &newBuildInput, OptixVertexFormat vFormat,
    unsigned int vStride, OptixIndicesFormat iFormat, unsigned int iStride,
    unsigned int sSize = 0, unsigned int sStride = 0)
{
    auto &info = newBuildInput.triangleArray;
    info.vertexFormat = vFormat;
    info.vertexStrideInBytes = vStride;
    info.indexFormat = iFormat;
    info.indexStrideInBytes = iStride;
    info.sbtIndexOffsetSizeInBytes = sSize;
    info.sbtIndexOffsetStrideInBytes = sStride;
}

/// @brief Fill the build input with triangle data; notice that stride and
/// format isn't set here.
/// @param[out] newBuildInput
/// @param newData triangle data buffer, used to fill vertex buffer, index
/// buffer and flag buffer of build input.
/// @param vertNum number of vertices
/// @param triNum number of triangles
static void FillBuildInputByTriangleDataBuffer(
    OptixBuildInput &newBuildInput, const TriangleDataBuffer &newData,
    TriangleVertexNumType vertNum, TriangleIndexNumType triNum)
{
    auto &info = newBuildInput.triangleArray;
    info = {};
    info.vertexBuffers = newData.GetVertexPtrsPtr();
    info.numVertices = vertNum;
    info.indexBuffer = newData.GetTrianglesPtr();
    info.numIndexTriplets = triNum;
    // TODO: here optix seems to have stricter requirements?
    HostUtils::CheckError(
        HostUtils::CheckInRangeAndSet(newData.GetFlagNum(), info.numSbtRecords),
        "Too many sbt records, it should be within unsigned int.");
    info.flags = newData.GetFlagPtr();
    info.sbtIndexOffsetBuffer = newData.GetSBTIndexOffsetBuffer();
}

/// @brief Construct triangle buffer; all vertices will be allocated on the same
/// buffer with proper alignment.
/// @param verticesArr array of vertex span
/// @param triangles span of triangle index; copied by value, so not expect a
/// owning type like vector.
/// @param flags span of flags or a single flag; copied by value.
TriangleDataBuffer::TriangleDataBuffer(const auto &verticesArr, auto triangles,
                                       auto flags)
    : flagBuffer_{ flags }
{
    auto singleSize = verticesArr[0].size_bytes();
    auto motionKeyNum = verticesArr.size();

    constexpr std::size_t c_optixRecommendAlign = 16;
    std::size_t roundedSize =
        UniUtils::RoundUpNonNegative(singleSize, c_optixRecommendAlign);

    assert((std::numeric_limits<std::size_t>::max)() / motionKeyNum >=
           roundedSize);
    std::size_t totalSize = roundedSize * motionKeyNum;
    verticesBuffer_ =
        HostUtils::DeviceMakeUninitializedUnique<std::byte[]>(totalSize);

    vertexPtrArr_ = std::make_unique<CUdeviceptr[]>(motionKeyNum);
    for (std::size_t i = 0; i < motionKeyNum; i++)
    {
        auto byteArr = std::as_bytes(verticesArr[i]);
        auto dstPtr = verticesBuffer_.get() + roundedSize * i;
        thrust::copy_n(byteArr.data(), byteArr.size(), dstPtr);
        vertexPtrArr_[i] = HostUtils::ToDriverPointer(dstPtr);
    }

    auto triangleByteArr = std::as_bytes(triangles);
    trianglesBuffer_ =
        HostUtils::DeviceMakeUnique<std::byte[]>(triangleByteArr);
    motionKeyNum_ = motionKeyNum;
}

/// @brief Construct triangle buffer; all vertices will be allocated on the same
/// buffer with proper alignment.
/// @param verticesArr array of vertex span
/// @param triangles span of triangle index; copied by value, so not expect a
/// owning type like vector.
/// @param flags span of flags or a single flag; copied by value.
/// @param sbtOffsets span of sbt offsets; copied by value.
TriangleDataBuffer::TriangleDataBuffer(const auto &verticesArr, auto triangles,
                                       auto flags, auto sbtOffsets)
    : TriangleDataBuffer{ verticesArr, triangles, flags }
{
    HostUtils::CheckError(sbtOffsets.size() == triangles.size() / 3,
                          "SBT offset should match the size of primitives.");
    sbtIndexOffsetBuffer_ =
        HostUtils::DeviceMakeUnique<std::byte[]>(std::as_bytes(sbtOffsets));
}

/// @brief Add build input when the triangle data buffer is pushed back in
/// handle. When an exception is thrown, the data buffer will be popped up, so
/// that number of build input and data buffer matches and memory is released
/// timely. This is the source of exception guarantee.
/// @param handle providing newly-pushed build input, handle should fill it and
/// push a new data buffer correspondingly.
void TriangleBuildInputArray::GeneralAddBuildInput_(auto &&handle)
{
    auto &newBuildInput = buildInputs_.emplace_back();
    try
    {
        handle(newBuildInput);
    }
    catch (...)
    {
        buildInputs_.pop_back(); // maintain the same number of build input and
                                 // data buffer.
        // NOTE: this segment can be omitted if nothing after .emplace_back in
        // try block may throw.
        if (buildInputs_.size() != dataBuffers_.size())
            dataBuffers_.pop_back();
        throw;
    }
    return;
}

void TriangleBuildInputArray::AddBuildInput(
    const std::vector<std::span<const float>> &vertices,
    std::span<const int> triangles, GeometryFlags flag)
{
    GeneralAddBuildInput_([&, this](OptixBuildInput &newBuildInput) {
        auto [vertNum, triNum] =
            PrepareTriangleBuildInput(newBuildInput, vertices, triangles);
        auto &newData = dataBuffers_.emplace_back(vertices, triangles, flag);
        FillBuildInputByTriangleDataBuffer(newBuildInput, newData, vertNum,
                                           triNum);
        FillTriangleBuildInputFormatAndStrides(
            newBuildInput, OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3, 0,
            OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3, 0);
    });
}

void TriangleBuildInputArray::AddBuildInput(
    const std::vector<std::span<const float>> &vertices,
    std::span<const int> triangles, std::span<GeometryFlags> flags,
    std::span<const std::uint32_t> sbtIndexOffset)
{
    GeneralAddBuildInput_([&, this](OptixBuildInput &newBuildInput) {
        auto [vertNum, triNum] =
            PrepareTriangleBuildInput(newBuildInput, vertices, triangles);
        auto &newData = dataBuffers_.emplace_back(vertices, triangles, flags,
                                                  sbtIndexOffset);
        FillBuildInputByTriangleDataBuffer(newBuildInput, newData, vertNum,
                                           triNum);
        FillTriangleBuildInputFormatAndStrides(
            newBuildInput, OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3, 0,
            OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3, 0,
            sizeof(std::uint32_t), 0);
    });
}

void TriangleBuildInputArray::RemoveBuildInput(std::size_t idx) noexcept
{
    BuildInputArray::RemoveBuildInput(idx);
    dataBuffers_.erase(dataBuffers_.begin() + idx);
}

InstanceDataBuffer::InstanceDataBuffer(auto instances)
    : instancesBuffer_{ HostUtils::DeviceMakeUnique<OptixInstance[]>(
          instances) }
{
}

using InstanceNumType = decltype(OptixBuildInputInstanceArray::numInstances);

static void FillBuildInputByInstanceDataBuffer(
    InstanceNumType instanceNum, OptixBuildInput &newBuildInput,
    const InstanceDataBuffer &newData)
{
    auto &info = newBuildInput.instanceArray;
    info.instances = newData.GetInstanceBufferPtr();
    info.numInstances = instanceNum;
#if OPTIX_VERSION >= 80000
    info.instanceStride = 0;
#endif
}

void InstanceBuildInputArray::AddBuildInput(
    std::span<OptixInstance> instances, std::span<const Traversable *> children)
{
    auto size = instances.size();
    HostUtils::CheckError(size == children.size(),
                          "Number of instances and number of children should "
                          "be the same.");
    InstanceNumType safeSize;
    HostUtils::CheckError(HostUtils::CheckInRangeAndSet(size, safeSize),
                          "Too many instances to build a build input.");

    for (std::size_t i = 0; i < size; i++)
        instances[i].traversableHandle = children[i]->GetHandle();

    auto &newBuildInput = buildInputs_.emplace_back();
    try
    {
        auto &newData = dataBuffers_.emplace_back(instances);
        FillBuildInputByInstanceDataBuffer(safeSize, newBuildInput, newData);
        children_.push_back(children | std::ranges::to<std::vector>());
    }
    catch (...)
    {
        buildInputs_.pop_back(); // maintain the same number of build input and
                                 // data buffer.
        if (buildInputs_.size() != dataBuffers_.size())
            dataBuffers_.pop_back();
        throw;
    }
}

void InstanceBuildInputArray::RemoveBuildInput(std::size_t idx) noexcept
{
    BuildInputArray::RemoveBuildInput(idx);
    dataBuffers_.erase(dataBuffers_.begin() + idx);
    children_.erase(children_.begin() + idx);
}

/// @brief Traverse all children to get the deepest depth.
unsigned int InstanceBuildInputArray::GetDepth() const noexcept
{
    return 1 +
           std::ranges::max(
               children_ |
               std::views::transform([](const auto &singleChildren) {
                   // For each instance group, get the maximum depth;
                   return std::ranges::max(
                       singleChildren |
                       std::views::transform([](const Traversable *childPtr) {
                           return childPtr->GetDepth();
                       }));
               }));
}

} // namespace Wayland::Optix