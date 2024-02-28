#pragma once

#include <concepts>
#include <limits>
#include <utility>

namespace HostUtils
{

template<std::integral DstType, std::integral SrcType>
inline bool CheckInRange(SrcType src, DstType &dst,
                         DstType min = (std::numeric_limits<DstType>::min)(),
                         DstType max = (std::numeric_limits<DstType>::max)())
{
    dst = static_cast<DstType>(src);
    return std::cmp_less_equal(min, dst) && std::cmp_less_equal(dst, max);
}

decltype(auto) Access(auto &&rng, auto &&key)
{
#ifdef NEED_IN_RANGE_CHECK
    return rng.at(key);
#else
    return rng[key];
#endif
}

} // namespace HostUtils
