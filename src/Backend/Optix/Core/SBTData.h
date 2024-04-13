#pragma once
#include "HostUtils/CommonHeaders.h"
#include <iterator>

namespace EasyRender::Optix
{

/// @brief template class for data passed to SBT.
template<typename T>
    requires std::is_trivially_copyable_v<T> || std::is_same_v<T, void>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SBTData
{
    std::byte header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;

    using value_type = T;
};

/// @brief specialized class to denote empty case (i.e. no data).
template<>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SBTData<void>
{
    std::byte header[OPTIX_SBT_RECORD_HEADER_SIZE];

    using value_type = void;
};

template<typename T>
concept IsSBTData =
    requires { std::is_same_v<SBTData<typename T::value_type>, T>; };

template<typename T>
concept IsSBTDataContiguousRange = requires(T x) {
    requires IsSBTData<std::remove_cvref_t<decltype(*std::ranges::data(x))>>;
};

} // namespace EasyRender::Optix