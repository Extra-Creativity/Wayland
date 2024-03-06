#pragma once
#include "HostUtils/CommonHeaders.h"

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

} // namespace Wayland::Optix