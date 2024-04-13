#include "ShaderBindingTable.h"

#include "HostUtils/ErrorCheck.h"
#include "ProgramGroup.h"
#include "UniUtils/IntUtils.h"

#include <algorithm>
#include <ranges>

using namespace EasyRender;

namespace EasyRender::Optix
{

void ShaderBindingTable::SetSBTHeader_(void *ptr,
                                       const ProgramGroupArray &programGroups,
                                       std::size_t idx)
{
    HostUtils::CheckOptixError(optixSbtRecordPackHeader(
        HostUtils::Access(programGroups.GetHandleArr(), idx), ptr));
}

/// @brief Copy all things to a single buffer with proper alignments.
/// @param[in, out] copyInfos host buffer and pointer of destination;
/// destination will be set to the address of device buffers.
void ShaderBindingTable::CopyToBuffer_(
    std::initializer_list<CopyInfo> copyInfos)
{
    constexpr std::size_t requiredAlignment = alignof(SBTData<void>);

    auto totalSize = std::ranges::fold_left(
        copyInfos | std::views::transform([](const auto &info) noexcept {
            return UniUtils::RoundUpNonNegative(info.hostBuffer.size(),
                                                requiredAlignment);
        }),
        0, std::plus<std::size_t>{});
    buffer_ = HostUtils::DeviceMakeUninitializedUnique<std::byte[]>(totalSize);

    std::size_t offset = 0;
    for (const auto &[hostBuffer, dst] : copyInfos)
    {
        auto dstPtr = buffer_.get() + offset;
        assert(reinterpret_cast<std::uintptr_t>(dstPtr.get()) %
                   OPTIX_SBT_RECORD_ALIGNMENT ==
               0);

        auto size = hostBuffer.size();
        thrust::copy_n(hostBuffer.data(), size, dstPtr);
        *dst = HostUtils::ToDriverPointer(dstPtr);
        offset += UniUtils::RoundUpNonNegative(size, requiredAlignment);
    }
    return;
}

} // namespace EasyRender::Optix