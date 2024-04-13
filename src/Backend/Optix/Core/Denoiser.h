#pragma once

#include "HostUtils/CommonHeaders.h"
#include "HostUtils/DebugUtils.h"
#include "HostUtils/DeviceAllocators.h"
#include "HostUtils/EnumUtils.h"
#include <concepts>
#include <optional>

namespace Wayland::Optix
{

/// @brief Specify the mode of denoiser; currently only NoIntensity,
/// NoAverageColor is usable for users, others are used to store states.
enum class Mode : std::uint32_t
{
    None = 0,
    // Modes below can be changed without initiating a new denoiser.

    /// Doesn't need to calculate average intensity of the input; this is
    /// enabled in LDR since average intensity is only useful in HDR.
    NoIntensity = 1,
    /// Doesn't need to calculate average color of the input; this is
    /// enabled in LDR since average color is only useful in HDR.
    NoAverageColor = 2,
    // Modes below shouldn't be used for users; only used to record status.
    ClearSystemMode = 3, // All redundant modes will be cleared.
    Tiling = 4,
    Upscale2x = 8,
    Temporal = 16,
    TemporalNotFirstFrame = 32,
    LDR = 64
};

} // namespace Wayland::Optix

ENABLE_BINARY_OP_FOR_SCOPED_ENUM(Wayland::Optix::Mode)

namespace Wayland::Optix
{
/// @brief Specify the input and output of the denoiser; if your image is
/// tightly packed, you can use SetTightOptixImage2D to set it.
/// @see #SetTightOptixImage2D.
/// @note Though OptixImage2D needs `data` of `CUdeviceptr` type, it's allowed
/// to give either `nullptr` or a host (CPU) pointer. Notice that `CUdeviceptr`
/// is an integer instead of `void*`, so you may need `reinterpret_cast`.
/// @note + If `nullptr` is provided, an device buffer will be allocated
/// automatically (uninitialized, no memset);
/// @note + If a host pointer is provided, an device buffer will also be
/// allocated, and the data will be copied to the device buffer (thus be sure
/// that host pointer points to valid memory in at least `rowStride * height`).
/// @note These two cases will preserve the buffer in the denoiser, which will
/// be freed when the denoiser is destructed; thus, it's unnecessary to worry
/// about memory leak.
/// @note + If a device pointer is provided, nothing will happen and we'll use
/// it directly in the denoiser. Thus be sure that pointer is valid during the
/// denoiser is used. ** Special notice that buffer of input and output cannot
/// be the same! **
struct ImageIOPair
{
    OptixImage2D input;
    OptixImage2D output;
};

/// @brief Contains a single pair, which denotes the input noisy image and the
/// output denoised image. ** It's necessary to provide width and height for
/// both input and output. **
struct ColorInfo
{
    ImageIOPair imageInfo;

    void CheckValid() const;
    const auto &GetColorInfo() const noexcept { return imageInfo; }
    auto GetMaxWidth() const noexcept { return imageInfo.input.width; }
    auto GetMaxHeight() const noexcept { return imageInfo.input.height; }
};

/// @brief Contains any pair of images, like colors, normals, etc.. **It's
/// necessary to make the first image to be color image.**
/// @note `width` and `height` are also provided; It's not necessary to set all
/// width & height in `OptixImage2D`, and we'll set them to the provided width
/// & height; but if you specify them as non-zero, we'll leave them alone.
struct AOVInfo
{
    unsigned int width;
    unsigned int height;
    std::vector<ImageIOPair> imageInfos;

    void CheckValid() const;
    const auto &GetColorInfo() const noexcept { return imageInfos[0]; }
    unsigned int GetMaxWidth() const noexcept;
    unsigned int GetMaxHeight() const noexcept;
};

/// @brief Temporal information; in temporal mode, 2D flow vector needs to be
/// provided, and the trustworthiness (1D) is optional. If you don't have flow
/// vector but still want to use temporal mode to utilize the last frame, you
/// may set them to be both buffer that's full of zero. But we don't know
/// whether it's really useful; this is determined by NVIDIA AI denoiser model,
/// which isn't open-source and lacks doc.
struct FlowInfo
{
    OptixImage2D flow;
    std::optional<OptixImage2D> flowTrust;
};

/// @brief Guide information at camera space; you can provide albedo and normal.
/// Notice that in Optix SDK, these two are required to be all or none, i.e. you
/// cannot provide only albedo or normal; we're not sure whether this is
/// expected, so we don't provide optional one.
/// @note if the data pointer is
struct GuideInfo
{
    OptixImage2D albedo;
    OptixImage2D normal;
};

/// @brief If you hope to denoiser by tiles, you need to provide a width and
/// height; it will be set to width and height of the image it's greater
/// than that or is 0.
struct TileInfo
{
    unsigned int tiledWidth;
    unsigned int tiledHeight;
};

struct BasicInfo
{
    Mode mode = Mode::None;
    OptixDenoiserAlphaMode alphaMode = OPTIX_DENOISER_ALPHA_MODE_COPY;
};

// Base class of all denoisers, which cannot be created directly by users.
class Denoiser
{
    using DenoiserDeleter = decltype([](OptixDenoiser denoiser) {
        HostUtils::CheckOptixError<HostUtils::OnlyLog>(
            optixDenoiserDestroy(denoiser));
    });
    using DenoiserWrapper = std::unique_ptr<OptixDenoiser_t, DenoiserDeleter>;

public:
    /// @param blendFactor range should be [0, 1]; 0 means use denoised version
    /// totally, and 1 means use the noisy version totally. Default to be 0.
    /// @param offsetX where to begin to denoise in the image at X axis (i.e.
    /// column of the image), default 0.
    /// @param offsetY where to begin to denoise in the image at Y axis (i.e.
    /// row of the image), default 0.
    void Denoise(float blendFactor = 0.0f, unsigned int offsetX = 0,
                 unsigned int offsetY = 0);
    virtual ~Denoiser() = default;

    bool NeedIntensity() const noexcept
    {
        return !HostUtils::TestEnum(mode_, Mode::NoIntensity);
    }

    bool NeedAverageColor() const noexcept
    {
        return !HostUtils::TestEnum(mode_, Mode::NoAverageColor);
    }

    /// @brief Return the pointer of output image; if device buffer is
    /// provided before in the OptixImage2D, it should be the provided buffer.
    void *GetOutputDataPtr(std::size_t idx = 0) const
    {
        return reinterpret_cast<void *>(
            HostUtils::Access(layers_, idx).output.data);
    }

    /// @brief Return the pointer of input image; if device buffer is
    /// provided before in the OptixImage2D, it should be the provided buffer.
    void *GetInputDataPtr(std::size_t idx = 0) const
    {
        return reinterpret_cast<void *>(
            HostUtils::Access(layers_, idx).input.data);
    }

    // TODO: virtual Resize.

protected:
    Denoiser(OptixDenoiserModelKind, const BasicInfo &, const GuideInfo *);

    void SetupDenoiser_(const TileInfo *, unsigned int, unsigned int);
    std::size_t SetTemporalGuideInputs_(const FlowInfo &);
    void SetTemporalGuideOutputs_(std::size_t);

    void TryAllocOnDevice_(OptixImage2D &);
    void TrySetOtherGuides_(const GuideInfo *, unsigned int, unsigned int);

    void PrepareColorLayer_(const ImageIOPair &);
    void PrepareColorLayer_(const ImageIOPair &, unsigned int, unsigned int);
    template<bool>
    void PrepareAOVLayers_(const AOVInfo &);
    void PrepareTemporalColorLayer_(const ImageIOPair &, const FlowInfo &);
    void PrepareTemporalColorLayer_(const ImageIOPair &, const FlowInfo &,
                                    unsigned int, unsigned int);

    CUdeviceptr GetScratchBufferPtr_() const noexcept;
    CUdeviceptr GetStateBufferPtr_() const noexcept;

protected:
    Mode mode_;

    // The size will be changed only when input size is changed.
    Wayland::HostUtils::DeviceUniquePtr<std::byte[]> buffer_;
    std::size_t stateBufferSize_ = 0;
    std::size_t scratchBufferSize_ = 0;

    // The size will be changed only when mode is changed.
    Wayland::HostUtils::DeviceUniquePtr<float[]> assistBuffer_;

    OptixDenoiserGuideLayer guideLayer_{};
    // The size will be changed when the input or output size is changed.
    Wayland::HostUtils::DeviceUniquePtr<std::byte[]> guideLayerTemporalBuffer_;
    std::size_t guideLayerTemporalBufferSize_ = 0;
    Wayland::HostUtils::DeviceUniquePtr<std::byte[]> guideLayerOrdinaryBuffer_;
    std::size_t guideLayerOrdinaryBufferSize_ = 0;

    std::vector<OptixDenoiserLayer> layers_;
    // The size will be changed when the output size is changed.
    std::vector<Wayland::HostUtils::DeviceUniquePtr<std::byte[]>> layerBuffers_;

    DenoiserWrapper denoiser_;
};

template<OptixDenoiserModelKind Kind>
class BasicDenoiser : public Denoiser
{
public:
    BasicDenoiser(const BasicInfo &, const ColorInfo &,
                  const TileInfo * = nullptr, const GuideInfo * = nullptr);
};

using LDRDenoiser = BasicDenoiser<OPTIX_DENOISER_MODEL_KIND_LDR>;
using HDRDenoiser = BasicDenoiser<OPTIX_DENOISER_MODEL_KIND_HDR>;

class TemporalDenoiser : public Denoiser
{
public:
    TemporalDenoiser(const BasicInfo &, const ColorInfo &, const FlowInfo &,
                     const TileInfo * = nullptr, const GuideInfo * = nullptr);
};

template<OptixDenoiserModelKind Kind>
class AOVBasicDenoiser : public Denoiser
{
public:
    AOVBasicDenoiser(const BasicInfo &, const AOVInfo &,
                     const TileInfo * = nullptr, const GuideInfo * = nullptr);
};

using AOVDenoiser = AOVBasicDenoiser<OPTIX_DENOISER_MODEL_KIND_AOV>;
using UpscaleDenoiser = AOVBasicDenoiser<OPTIX_DENOISER_MODEL_KIND_UPSCALE2X>;

template<OptixDenoiserModelKind Kind>
class AOVTemporalBasicDenoiser : public Denoiser
{
public:
    AOVTemporalBasicDenoiser(const BasicInfo &, const AOVInfo &,
                             const FlowInfo &, const TileInfo * = nullptr,
                             const GuideInfo * = nullptr);
};

using AOVTemporalDenoiser =
    AOVTemporalBasicDenoiser<OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV>;
using UpscaleTemporalDenoiser =
    AOVTemporalBasicDenoiser<OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X>;

} // namespace Wayland::Optix
