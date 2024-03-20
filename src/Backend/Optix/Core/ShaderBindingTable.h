#pragma once
#include "HostUtils/CommonHeaders.h"
#include "HostUtils/DeviceAllocators.h"

namespace Wayland::Optix
{

/// @brief template class for data passed to SBT.
template<typename T>
    requires std::is_trivially_copyable_v<T> || std::is_same_v<T, void>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SBTData
{
    std::byte header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

/// @brief specialized class to denote empty case (i.e. no data).
template<>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SBTData<void>
{
    std::byte header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

class ProgramGroupArray;
/// @brief Abstraction of SBT; used to create the SBT from the data passed to it
/// and manage them.
class ShaderBindingTable
{
    static void SetSBTHeader_(void *ptr, const ProgramGroupArray &programGroups,
                              std::size_t idx);

    struct CopyInfo
    {
        std::span<const std::byte> hostBuffer;
        CUdeviceptr *devicePtr;
    };

    void CopyToBuffer_(std::initializer_list<CopyInfo> hostBuffers);

public:
    /// @brief Construct SBT using SBT data; SBT data should be non-const
    /// because its header will be set according to program groups.
    /// @param raygenData
    /// @param raygenIdx index of raygen program in the program group array.
    /// @param missData span of data of miss programs, since it can be more than
    /// one.
    /// @param missIdx pointer of indices of miss program in the program array;
    /// missIdx[i] should be valid for any i in [0, missData.size()).
    /// @param hitData span of data of hit programs.
    /// @param hitIdx pointer of indices of hit groupsprogram in the program
    /// array; hitIdx[i] should be valid for any i in [0, hitData.size()).
    /// @param programGroups
    template<typename RaygenData, typename MissData, typename HitData>
    ShaderBindingTable(SBTData<RaygenData> &raygenData, std::size_t raygenIdx,
                       const std::span<SBTData<MissData>> missData,
                       const std::size_t *missIdx,
                       const std::span<SBTData<HitData>> hitData,
                       const std::size_t *hitIdx,
                       const ProgramGroupArray &programGroups)
    {
        // Not included DebugUtils to use CheckInRange since we only do it once.
        assert(missData.size() <= (std::numeric_limits<unsigned int>::max)() &&
               hitData.size() <= (std::numeric_limits<unsigned int>::max)());
        SetSBTHeader_(&raygenData, programGroups, raygenIdx);
        for (std::size_t i = 0; i < missData.size(); i++)
            SetSBTHeader_(missData.data() + i, programGroups, missIdx[i]);
        for (std::size_t i = 0; i < hitData.size(); i++)
            SetSBTHeader_(hitData.data() + i, programGroups, hitIdx[i]);

        CopyToBuffer_({
            CopyInfo{
                .hostBuffer = { (std::byte *)(&raygenData),
                                sizeof(raygenData) },
                .devicePtr = &sbt_.raygenRecord,
            },
            { std::as_bytes(missData), &sbt_.missRecordBase },
            { std::as_bytes(hitData), &sbt_.hitgroupRecordBase },
        });
        sbt_.missRecordCount = (unsigned int)missData.size();
        sbt_.missRecordStrideInBytes = { sizeof(missData[0]) };
        sbt_.hitgroupRecordCount = (unsigned int)hitData.size();
        sbt_.hitgroupRecordStrideInBytes = { sizeof(hitData[0]) };
    }

    /// @brief same as ctor above; the miss program is restricted to be one so
    /// that index is passed directly.
    template<typename RaygenData, typename MissData, typename HitData>
    ShaderBindingTable(SBTData<RaygenData> &raygenData, std::size_t raygenIdx,
                       SBTData<MissData> &missData, std::size_t missIdx,
                       const std::span<SBTData<HitData>> hitData,
                       const std::size_t *hitIdx,
                       const ProgramGroupArray &programGroups)
    {
        assert(hitData.size() <= (std::numeric_limits<unsigned int>::max)());
        SetSBTHeader_(&raygenData, programGroups, raygenIdx);
        SetSBTHeader_(&missData, programGroups, missIdx);
        for (std::size_t i = 0; i < hitData.size(); i++)
            SetSBTHeader_(hitData.data() + i, programGroups, hitIdx[i]);

        CopyToBuffer_({
            CopyInfo{
                .hostBuffer = { (std::byte *)(&raygenData),
                                sizeof(raygenData) },
                .devicePtr = &sbt_.raygenRecord,
            },
            CopyInfo{
                .hostBuffer = { (std::byte *)(&missData), sizeof(missData) },
                .devicePtr = &sbt_.missRecordBase,
            },
            { std::as_bytes(hitData), &sbt_.hitgroupRecordBase },
        });
        sbt_.missRecordCount = 1;
        sbt_.missRecordStrideInBytes = sizeof(missData);
        sbt_.hitgroupRecordCount = (unsigned int)hitData.size();
        sbt_.hitgroupRecordStrideInBytes = { sizeof(hitData[0]) };
    }

    const auto &GetHandle() const noexcept { return sbt_; }

private:
    Wayland::HostUtils::DeviceUniquePtr<std::byte[]> buffer_;
    OptixShaderBindingTable sbt_{};
};

} // namespace Wayland::Optix