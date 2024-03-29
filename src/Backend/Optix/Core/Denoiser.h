#pragma once

#include "HostUtils/DebugUtils.h"
#include "HostUtils/DeviceAllocators.h"
#include "HostUtils/EnumUtils.h"
#include <concepts>
#include <optional>

namespace Wayland::Optix
{

enum class Mode : std::uint32_t
{
    None = 0,
    Tiling = 1,
    Upscale2x = 2,
    // Modes below can be changed without initiating a new denoiser.
    NoIntensity = 4,
    NoAverageColor = 8,
    // Modes below shouldn't be used for users.
    ClearSystemMode = 15,
    Temporal = 16,
    TemporalNotFirstFrame = 32,
    LDR = 64
};

} // namespace Wayland::Optix

ENABLE_BINARY_OP_FOR_SCOPED_ENUM(Wayland::Optix::Mode)

namespace Wayland::Optix
{
struct ImageIOPair
{
    OptixImage2D input;
    OptixImage2D output;
};

struct ColorInfo
{
    ImageIOPair imageInfo;

    void CheckValid() const;
    const auto &GetColorInfo() const noexcept { return imageInfo; }
    auto GetMaxWidth() const noexcept { return imageInfo.input.width; }
    auto GetMaxHeight() const noexcept { return imageInfo.input.height; }
};

// It's not necessary to set width & height in OptixImage2D; we'll set
// it to uniform width and height provided in AOVInfo. But if you specify it
// as non-zero, we'll leave it along.
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

struct FlowInfo
{
    OptixImage2D flow;
    std::optional<OptixImage2D> flowTrust;
};

struct GuideInfo
{
    OptixImage2D albedo;
    OptixImage2D normal;
};

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

class Denoiser
{
    using DenoiserDeleter = decltype([](OptixDenoiser denoiser) {
        HostUtils::CheckOptixError<HostUtils::OnlyLog>(
            optixDenoiserDestroy(denoiser));
    });
    using DenoiserWrapper = std::unique_ptr<OptixDenoiser_t, DenoiserDeleter>;

public:
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

    void *GetOutputDataPtr(std::size_t idx) const
    {
        return reinterpret_cast<void *>(
            HostUtils::Access(layers_, idx).output.data);
    }

    void *GetInputDataPtr(std::size_t idx) const
    {
        return reinterpret_cast<void *>(
            HostUtils::Access(layers_, idx).input.data);
    }

protected:
    Denoiser(OptixDenoiserModelKind, const BasicInfo &, const GuideInfo *);

    void SetupDenoiser_(const TileInfo *, unsigned int, unsigned int);
    std::size_t SetTemporalGuideInputs_(const FlowInfo &);
    void SetGuideOutputs_(std::size_t);

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
