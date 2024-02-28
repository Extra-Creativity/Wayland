#pragma once
#include "HostUtils/CommonHeaders.h"
#include "HostUtils/DeviceAllocators.h"

template<typename T>
    requires std::is_trivially_copyable_v<T> || std::is_same_v<T, void>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SBTData
{
    std::byte header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

template<>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SBTData<void>
{
    std::byte header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

class ProgramGroupArray;
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
    template<typename RaygenData, typename MissData, typename HitData>
    ShaderBindingTable(SBTData<RaygenData> &raygenData, std::size_t raygenIdx,
                       const std::span<SBTData<MissData>> missData,
                       const std::size_t *missIdx,
                       const std::span<SBTData<HitData>> hitData,
                       const std::size_t *hitIdx,
                       const ProgramGroupArray &programGroups)
    {
        assert(missData.size() <= (std::numeric_limits<unsigned int>::max)() &&
               hitData.size() <= (std::numeric_limits<unsigned int>::max)());
        SetSBTHeader_(&raygenData, programGroups, raygenIdx);
        for (std::size_t i = 0; i < missData.size(); i++)
            SetSBTHeader_(missData.data() + i, programGroups, missIdx[i]);
        for (std::size_t i = 0; i < hitData.size(); i++)
            SetSBTHeader_(hitData.data() + i, programGroups, hitIdx[i]);

        CopyToBuffer_(
            { CopyInfo{
                  .hostBuffer = { reinterpret_cast<std::byte *>(&raygenData),
                                  sizeof(raygenData) },
                  .devicePtr = &sbt_.raygenRecord,
              },
              { std::as_bytes(missData), &sbt_.missRecordBase },
              { std::as_bytes(hitData), &sbt_.hitgroupRecordBase } });
        sbt_.missRecordCount = (unsigned int)missData.size();
        sbt_.missRecordStrideInBytes = { sizeof(missData[0]) };
        sbt_.hitgroupRecordCount = (unsigned int)hitData.size();
        sbt_.hitgroupRecordStrideInBytes = { sizeof(hitData[0]) };
    }

    const auto &GetHandle() const noexcept { return sbt_; }

private:
    HostUtils::DeviceUniquePtr<std::byte[]> buffer_;
    OptixShaderBindingTable sbt_{};
};