#pragma once
#include "HostUtils/CommonHeaders.h"
#include "HostUtils/DeviceAllocators.h"

#include "SBTData.h"

namespace EasyRender::Optix
{

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
        requires IsSBTDataContiguousRange<MissData> &&
                 IsSBTDataContiguousRange<HitData>
    ShaderBindingTable(SBTData<RaygenData> &raygenData, std::size_t raygenIdx,
                       MissData &missData, const std::size_t *missIdx,
                       HitData &hitData, const std::size_t *hitIdx,
                       const ProgramGroupArray &programGroups)
    {
        // Not included DebugUtils to use CheckInRange since we only do it once.
        std::span missSpan{ std::ranges::data(missData),
                            std::ranges::size(missData) };
        std::span hitSpan{ std::ranges::data(hitData), std::ranges::size(hitData) };
        assert(missSpan.size() <= (std::numeric_limits<unsigned int>::max)() &&
               hitSpan.size() <= (std::numeric_limits<unsigned int>::max)());

        SetSBTHeader_(&raygenData, programGroups, raygenIdx);
        for (std::size_t i = 0; i < missSpan.size(); i++)
            SetSBTHeader_(missSpan.data() + i, programGroups, missIdx[i]);
        for (std::size_t i = 0; i < hitSpan.size(); i++)
            SetSBTHeader_(hitSpan.data() + i, programGroups, hitIdx[i]);

        CopyToBuffer_({
            CopyInfo{
                .hostBuffer = { (std::byte *)(&raygenData),
                                sizeof(raygenData) },
                .devicePtr = &sbt_.raygenRecord,
            },
            { std::as_bytes(missSpan), &sbt_.missRecordBase },
            { std::as_bytes(hitSpan), &sbt_.hitgroupRecordBase },
        });
        sbt_.missRecordCount = (unsigned int)missSpan.size();
        sbt_.missRecordStrideInBytes = { sizeof(missSpan[0]) };
        sbt_.hitgroupRecordCount = (unsigned int)hitSpan.size();
        sbt_.hitgroupRecordStrideInBytes = { sizeof(hitSpan[0]) };
    }

    /// @brief same as ctor above; the miss program is restricted to be one so
    /// that index is passed directly.
    template<typename RaygenData, typename MissData, typename HitData>
        requires IsSBTDataContiguousRange<HitData>
    ShaderBindingTable(SBTData<RaygenData> &raygenData, std::size_t raygenIdx,
                       SBTData<MissData> &missData, std::size_t missIdx,
                       HitData &hitData, const std::size_t *hitIdx,
                       const ProgramGroupArray &programGroups)
    {
        std::span hitSpan{ std::ranges::data(hitData),
                           std::ranges::size(hitData) };
        assert(hitSpan.size() <= (std::numeric_limits<unsigned int>::max)());

        SetSBTHeader_(&raygenData, programGroups, raygenIdx);
        SetSBTHeader_(&missData, programGroups, missIdx);
        for (std::size_t i = 0; i < hitSpan.size(); i++)
            SetSBTHeader_(hitSpan.data() + i, programGroups, hitIdx[i]);

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
            { std::as_bytes(hitSpan), &sbt_.hitgroupRecordBase },
        });
        sbt_.missRecordCount = 1;
        sbt_.missRecordStrideInBytes = sizeof(missData);
        sbt_.hitgroupRecordCount = (unsigned int)hitSpan.size();
        sbt_.hitgroupRecordStrideInBytes = { sizeof(hitSpan[0]) };
    }

    const auto &GetHandle() const noexcept { return sbt_; }

private:
    EasyRender::HostUtils::DeviceUniquePtr<std::byte[]> buffer_;
    OptixShaderBindingTable sbt_{};
};

} // namespace EasyRender::Optix