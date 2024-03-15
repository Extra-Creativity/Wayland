/*
Same requirements of C++20 headers <numbers>, but can be used in C++17.
*/
#pragma once

#include <type_traits>

namespace Wayland::DeviceUtils::Constants
{

template<class T>
struct RejectInvalid
{
    static_assert(std::is_floating_point_v<T>);
    using type = T;
};

template<class T>
using RejectInvalid_t = RejectInvalid<T>::type;

template<class T>
inline constexpr T e_v = static_cast<RejectInvalid_t<T>>(2.718281828459045);
template<class T>
inline constexpr T log2e_v =
    static_cast<RejectInvalid_t<T>>(1.4426950408889634);
template<class T>
inline constexpr T log10e_v =
    static_cast<RejectInvalid_t<T>>(0.4342944819032518);
template<class T>
inline constexpr T pi_v = static_cast<RejectInvalid_t<T>>(3.141592653589793);
template<class T>
inline constexpr T inv_pi_v =
    static_cast<RejectInvalid_t<T>>(0.3183098861837907);
template<class T>
inline constexpr T inv_sqrtpi_v =
    static_cast<RejectInvalid_t<T>>(0.5641895835477563);
template<class T>
inline constexpr T ln2_v = static_cast<RejectInvalid_t<T>>(0.6931471805599453);
template<class T>
inline constexpr T ln10_v = static_cast<RejectInvalid_t<T>>(2.302585092994046);
template<class T>
inline constexpr T sqrt2_v =
    static_cast<RejectInvalid_t<T>>(1.4142135623730951);
template<class T>
inline constexpr T sqrt3_v =
    static_cast<RejectInvalid_t<T>>(1.7320508075688772);
template<class T>
inline constexpr T inv_sqrt3_v =
    static_cast<RejectInvalid_t<T>>(0.5773502691896257);
template<class T>
inline constexpr T egamma_v =
    static_cast<RejectInvalid_t<T>>(0.5772156649015329);
template<class T>
inline constexpr T phi_v = static_cast<RejectInvalid_t<T>>(1.618033988749895);

inline constexpr float e = e_v<float>;
inline constexpr float log2e = log2e_v<float>;
inline constexpr float log10e = log10e_v<float>;
inline constexpr float pi = pi_v<float>;
inline constexpr float inv_pi = inv_pi_v<float>;
inline constexpr float inv_sqrtpi = inv_sqrtpi_v<float>;
inline constexpr float ln2 = ln2_v<float>;
inline constexpr float ln10 = ln10_v<float>;
inline constexpr float sqrt2 = sqrt2_v<float>;
inline constexpr float sqrt3 = sqrt3_v<float>;
inline constexpr float inv_sqrt3 = inv_sqrt3_v<float>;
inline constexpr float egamma = egamma_v<float>;
inline constexpr float phi = phi_v<float>;

} // namespace Wayland::DeviceUtils::Constants
