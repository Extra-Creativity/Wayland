#pragma once

#include <concepts>
#include <limits>
#include <utility>

namespace Wayland::HostUtils
{

template<std::integral DstType, std::integral SrcType>
inline bool CheckInRangeAndSet(
    SrcType src, DstType &dst,
    DstType min = (std::numeric_limits<DstType>::min)(),
    DstType max = (std::numeric_limits<DstType>::max)())
{
    dst = static_cast<DstType>(src);
    return std::cmp_less_equal(min, src) && std::cmp_less_equal(src, max);
}

template<std::integral DstType, std::integral SrcType>
inline bool CheckInRange(SrcType src,
                         DstType min = (std::numeric_limits<DstType>::min)(),
                         DstType max = (std::numeric_limits<DstType>::max)())
{
    return std::cmp_less_equal(min, src) && std::cmp_less_equal(src, max);
}

// See https://zhuanlan.zhihu.com/p/147039093.
template<std::integral SrcType>
inline bool CheckInRange(SrcType src, SrcType min, SrcType max)
{
    using UnsignedType = std::make_unsigned_t<SrcType>;
    using SignedType = std::make_signed_t<SrcType>;
    return ((SignedType)(((UnsignedType)src - (UnsignedType)min) |
                         ((UnsignedType)max - (UnsignedType)src))) >= 0;
}

decltype(auto) Access(auto &&rng, auto &&key)
{
#ifdef NEED_IN_RANGE_CHECK
    return rng.at(key);
#else
    return rng[key];
#endif
}

} // namespace Wayland::HostUtils
