#pragma once
#include "HostUtils/CommonHeaders.h"

namespace Wayland::Optix
{

/// @brief Wrapper of OPTIX_PIXEL_FORMAT
enum class PixelFormat
{
    Half1 = OPTIX_PIXEL_FORMAT_HALF1,
    Half2 = OPTIX_PIXEL_FORMAT_HALF2,
    Half3 = OPTIX_PIXEL_FORMAT_HALF3,
    Half4 = OPTIX_PIXEL_FORMAT_HALF4,
    Float1 = OPTIX_PIXEL_FORMAT_FLOAT1,
    Float2 = OPTIX_PIXEL_FORMAT_FLOAT2,
    Float3 = OPTIX_PIXEL_FORMAT_FLOAT3,
    Float4 = OPTIX_PIXEL_FORMAT_FLOAT4,
    UChar3 = OPTIX_PIXEL_FORMAT_UCHAR3,
    UChar4 = OPTIX_PIXEL_FORMAT_UCHAR4
};

namespace Details
{

// clang-format off
template<PixelFormat Format> struct FormatInfo { };
template<> struct FormatInfo<PixelFormat::Float1> { static constexpr std::size_t size = sizeof(float); };
template<> struct FormatInfo<PixelFormat::Float2> { static constexpr std::size_t size = sizeof(float) * 2; };
template<> struct FormatInfo<PixelFormat::Float3> { static constexpr std::size_t size = sizeof(float) * 3; };
template<> struct FormatInfo<PixelFormat::Float4> { static constexpr std::size_t size = sizeof(float) * 4; };
template<> struct FormatInfo<PixelFormat::Half1> { static constexpr std::size_t size = sizeof(float) / 2; };
template<> struct FormatInfo<PixelFormat::Half2> { static constexpr std::size_t size = sizeof(float); };
template<> struct FormatInfo<PixelFormat::Half3> { static constexpr std::size_t size = sizeof(float) / 2 * 3; };
template<> struct FormatInfo<PixelFormat::Half4> { static constexpr std::size_t size = sizeof(float) * 2; };
template<> struct FormatInfo<PixelFormat::UChar3> { static constexpr std::size_t size = sizeof(unsigned char) * 3; };
template<> struct FormatInfo<PixelFormat::UChar4> { static constexpr std::size_t size = sizeof(unsigned char) * 4; };

} // namespace Details

/// @brief Set all information of Image2D for a tightly packed image.
/// @tparam Format e.g. PixelFormat::Float4
/// @param image image to be set
/// @param data pointer to tightly packed image.
/// @param width 
/// @param height 
template<PixelFormat Format>
void SetTightOptixImage2D(OptixImage2D &image, void *data, unsigned int width,
                          unsigned int height)
{
    image.data = reinterpret_cast<CUdeviceptr>(data);
    image.width = width, image.height = height;
    image.rowStrideInBytes = Details::FormatInfo<Format>::size * width;
    image.pixelStrideInBytes = Details::FormatInfo<Format>::size;
    image.format = static_cast<decltype(image.format)>(Format);
}

} // namespace Wayland::Optix