#include "Denoiser.h"
#include "ContextManager.h"
#include "UniUtils/IntUtils.h"

#include <ranges>

using namespace EasyRender;

static OptixDenoiserOptions GetDenoiserOptions(OptixDenoiserAlphaMode mode,
                                               bool hasGuide)
{
    OptixDenoiserOptions options{};
    if (hasGuide)
        options.guideAlbedo = options.guideNormal = 1;
    options.denoiseAlpha = mode;
    return options;
}

/// @brief Copy from src to dst, and set the dst width or height if they're 0;
/// Unsafe because it doesn't check format of images.
/// @return Size of buffer that needs to be allocated; if it's already a device
/// buffer, return 0.
static std::size_t UnsafeCopyOptixImage(OptixImage2D &dstInfo,
                                        const OptixImage2D &srcInfo,
                                        unsigned int width, unsigned int height)
{
    dstInfo = srcInfo;

    if (srcInfo.width == 0)
        dstInfo.width = width;
    if (srcInfo.height == 0)
        dstInfo.height = height;

    if (HostUtils::IsFromDeviceMemory(reinterpret_cast<void *>(srcInfo.data)))
        return 0;
    return std::size_t{ srcInfo.rowStrideInBytes } * height;
}

static std::size_t SetFlow(OptixImage2D &dstFlow, const OptixImage2D &srcFlow,
                           unsigned int width, unsigned int height)
{
    HostUtils::CheckError(
        (srcFlow.format == OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT2) ||
            (srcFlow.format == OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF2),
        "Flow can only have two components.");
    return UnsafeCopyOptixImage(dstFlow, srcFlow, width, height);
}

static std::size_t SetFlowTrust(OptixImage2D &dstFlowTrust,
                                const OptixImage2D &srcFlowTrust,
                                unsigned int width, unsigned int height)
{
    HostUtils::CheckError(
        (srcFlowTrust.format == OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT1) ||
            (srcFlowTrust.format == OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF1),
        "Flow trustworthiness can only have a single component.");
    return UnsafeCopyOptixImage(dstFlowTrust, srcFlowTrust, width, height);
}

static std::size_t SetGuideNormal(OptixImage2D &dstNormal,
                                  const OptixImage2D &srcNormal,
                                  unsigned int width, unsigned int height)
{
    HostUtils::CheckError(
        (srcNormal.format == OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT3) ||
            (srcNormal.format == OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF3),
        "Normal can only have three components.");
    return UnsafeCopyOptixImage(dstNormal, srcNormal, width, height);
}

static std::size_t SetGuideAlbedo(OptixImage2D &dstAlbedo,
                                  const OptixImage2D &srcAlbedo,
                                  unsigned int width, unsigned int height)
{
    HostUtils::CheckError(
        (srcAlbedo.format == OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT1) ||
            (srcAlbedo.format == OptixPixelFormat::OPTIX_PIXEL_FORMAT_HALF1),
        "Albedo can only have a single component.");
    return UnsafeCopyOptixImage(dstAlbedo, srcAlbedo, width, height);
}

static void SetOutputInternalLayer(OptixImage2D &internalLayer,
                                   thrust::device_ptr<std::byte> ptr,
                                   unsigned width, unsigned int height)
{
    internalLayer.data = HostUtils::ToDriverPointer(ptr);
    internalLayer.width = width;
    internalLayer.height = height;
    internalLayer.rowStrideInBytes = internalLayer.pixelStrideInBytes * width;
    internalLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;
}

static std::size_t GetOutputInternalSize(const OptixDenoiserLayer &layer)
{
    static constexpr std::size_t internalAlignment = 16;
    return UniUtils::RoundUpNonNegative(std::size_t{ layer.output.width } *
                                            layer.output.height,
                                        internalAlignment);
}

static void UnifyImageSize(OptixDenoiserLayer &newLayer, unsigned int width,
                           unsigned int height)
{
    auto UnifySize = [](unsigned int &size, unsigned int expectSize) noexcept {
        if (size == 0)
            size = expectSize;
    };

    UnifySize(newLayer.input.width, width);
    UnifySize(newLayer.input.height, height);
    UnifySize(newLayer.output.width, newLayer.input.width);
    UnifySize(newLayer.output.height, newLayer.input.height);
}

static bool CheckValidColor(const OptixImage2D &image)
{
    return (image.format == OPTIX_PIXEL_FORMAT_FLOAT3) |
           (image.format == OPTIX_PIXEL_FORMAT_FLOAT4) |
           (image.format == OPTIX_PIXEL_FORMAT_HALF3) |
           (image.format == OPTIX_PIXEL_FORMAT_HALF4);
}

namespace EasyRender::Optix
{
/// @brief For a host pointer in data of OptixImage2D, if the size is not zero,
/// copy it to tPtr + offset and set the data to the driver pointer (if the
/// pointer is null, then memset to all 0).
/// @param tPtr the device buffer to be copied to.
/// @param offset offset from the beginning pointer (tPtr).
/// @param size size of the buffer to be copied.
/// @param[in,out] dst dst.data is the host buffer, and will be changed to the
/// device buffer.
static void CopyOrZero(auto tPtr, std::size_t offset, std::size_t size,
                       OptixImage2D &dst)
{
    using ElementType = decltype(*tPtr);
    if (size != 0)
    {
        tPtr += offset;
        auto dPtr = HostUtils::ToDriverPointer(tPtr);
        if (dst.data != 0)
            thrust::copy_n(reinterpret_cast<std::byte *>(dst.data), size, tPtr);
        else
            cudaMemset(tPtr.get(), 0, size);
        dst.data = dPtr;
    }
    return;
}

void ColorInfo::CheckValid() const
{
    HostUtils::CheckError(CheckValidColor(imageInfo.input) &&
                              CheckValidColor(imageInfo.output),
                          "Color image must contain 3 or 4 channels.");
    HostUtils::CheckError(
        imageInfo.input.width != 0 && imageInfo.input.height != 0 &&
            imageInfo.output.width != 0 && imageInfo.output.height != 0,
        "Pure color info should have width and height in both input and "
        "output.");
}

void AOVInfo::CheckValid() const
{
    HostUtils::CheckError(
        !imageInfos.empty(),
        "AOV should at least contains color image as the first variable.");

    // Required by optixDenoiseInvoke.
    HostUtils::CheckError(
        HostUtils::CheckInRange<unsigned int>(imageInfos.size()),
        "AOV number should be within limit of unsigned int.");

    HostUtils::CheckError(CheckValidColor(GetColorInfo().input) &&
                              CheckValidColor(GetColorInfo().output),
                          "Color image must contain 3 or 4 channels.");
}

unsigned int AOVInfo::GetMaxWidth() const noexcept
{
    return std::max(
        std::ranges::max(imageInfos |
                         std::views::transform([](const auto &info) {
                             return info.input.width;
                         })),
        width);
}

unsigned int AOVInfo::GetMaxHeight() const noexcept
{
    return std::max(
        std::ranges::max(imageInfos |
                         std::views::transform([](const auto &info) {
                             return info.input.height;
                         })),
        height);
}

Denoiser::Denoiser(OptixDenoiserModelKind kind, const BasicInfo &basicInfo,
                   const GuideInfo *guideInfo)
    : mode_{ basicInfo.mode & Mode::ClearSystemMode }
{
    OptixDenoiser rawDenoiser;
    auto options =
        GetDenoiserOptions(basicInfo.alphaMode, guideInfo != nullptr);
    optixDenoiserCreate(LocalContextSetter::GetCurrentOptixContext(), kind,
                        &options, &rawDenoiser);
    denoiser_.reset(rawDenoiser);
}

CUdeviceptr Denoiser::GetScratchBufferPtr_() const noexcept
{
    return HostUtils::ToDriverPointer(buffer_.get() + stateBufferSize_);
}

CUdeviceptr Denoiser::GetStateBufferPtr_() const noexcept
{
    return HostUtils::ToDriverPointer(buffer_.get());
}

void Denoiser::SetupDenoiser_(const TileInfo *tileInfo, unsigned int width,
                              unsigned int height)
{
    if (tileInfo)
    {
        if (tileInfo->tiledWidth != 0)
            width = std::min(tileInfo->tiledWidth, width);
        if (tileInfo->tiledHeight != 0)
            height = std::min(tileInfo->tiledHeight, height);
    }

    OptixDenoiserSizes sizeInfo;
    HostUtils::CheckOptixError(optixDenoiserComputeMemoryResources(
        denoiser_.get(), width, height, &sizeInfo));

    // This is guaranteed by optix.
    guideLayer_.outputInternalGuideLayer.pixelStrideInBytes =
        static_cast<unsigned int>(sizeInfo.internalGuideLayerPixelSizeInBytes);

    // Unfortunately, we don't know requirement of alignment, so we have to use
    // the most conservative one (i.e. same as cudaMalloc).
    static constexpr std::size_t alignment = 128;
    stateBufferSize_ =
        UniUtils::RoundUpNonNegative(sizeInfo.stateSizeInBytes, alignment);
    unsigned int overlap;
    if (tileInfo)
    {
        scratchBufferSize_ = sizeInfo.withOverlapScratchSizeInBytes;
        overlap = sizeInfo.overlapWindowSizeInPixels;
    }
    else
    {
        scratchBufferSize_ = sizeInfo.withoutOverlapScratchSizeInBytes;
        overlap = 0;
    }

    std::size_t totalSize = scratchBufferSize_ + stateBufferSize_;
    buffer_ = HostUtils::DeviceMakeUninitializedUnique<std::byte[]>(totalSize);

    std::size_t assistSize = 0;
    if (NeedIntensity())
        assistSize++;
    if (NeedAverageColor())
        assistSize += 3;
    if (assistSize != 0)
    {
        assistBuffer_ =
            HostUtils::DeviceMakeUninitializedUnique<float[]>(assistSize);
    }

    HostUtils::CheckOptixError(optixDenoiserSetup(
        denoiser_.get(), LocalContextSetter::GetCurrentCUDAStream(),
        width + 2 * overlap, height + 2 * overlap, GetStateBufferPtr_(),
        stateBufferSize_, GetScratchBufferPtr_(), scratchBufferSize_));
}

/// @brief Allocate guide buffer if it's available.
void Denoiser::TrySetOtherGuides_(const GuideInfo *guideInfo,
                                  unsigned int width, unsigned int height)
{
    if (guideInfo == nullptr)
        return;
    auto normalSize = SetGuideNormal(guideLayer_.normal, guideInfo->normal,
                                     width, height),
         albedoSize = SetGuideAlbedo(guideLayer_.albedo, guideInfo->albedo,
                                     width, height);
    constexpr std::size_t albedoAlignment = 128; // Conservative.
    auto alignedNormalSize =
        UniUtils::RoundUpNonNegative(normalSize, albedoAlignment);

    guideLayerOrdinaryBufferSize_ = alignedNormalSize + albedoSize;
    guideLayerOrdinaryBuffer_ =
        HostUtils::DeviceMakeUninitializedUnique<std::byte[]>(
            guideLayerOrdinaryBufferSize_);

    CopyOrZero(guideLayerOrdinaryBuffer_.get(), 0, normalSize,
               guideLayer_.normal);
    CopyOrZero(guideLayerOrdinaryBuffer_.get(), alignedNormalSize, normalSize,
               guideLayer_.albedo);
}

/// @brief Allocate layer buffer on the device if it's a host pointer;
/// layerBuffers_ will be pushed.
/// @param[in,out] image dst.data is the host buffer, and will be changed to the
/// device buffer.
[[nodiscard]] void Denoiser::TryAllocOnDevice_(OptixImage2D &image)
{
    if (HostUtils::IsFromDeviceMemory(reinterpret_cast<void *>(image.data)))
        return;

    std::size_t size = image.rowStrideInBytes * image.height;
    auto &buffer = layerBuffers_.emplace_back(
        HostUtils::DeviceMakeUninitializedUnique<std::byte[]>(size));
    if (image.data != 0)
        thrust::copy_n(reinterpret_cast<std::byte *>(image.data), size,
                       buffer.get());
    image.data = HostUtils::ToDriverPointer(buffer.get());
    return;
}

void Denoiser::PrepareColorLayer_(const ImageIOPair &info)
{
    auto &newLayer = layers_.emplace_back();
    newLayer.input = info.input, newLayer.output = info.output;
    TryAllocOnDevice_(newLayer.input);
    TryAllocOnDevice_(newLayer.output);
}

void Denoiser::PrepareColorLayer_(const ImageIOPair &info, unsigned int width,
                                  unsigned int height)
{
    auto &newLayer = layers_.emplace_back();
    newLayer.input = info.input, newLayer.output = info.output;
    UnifyImageSize(newLayer, width, height);
    TryAllocOnDevice_(newLayer.input);
    TryAllocOnDevice_(newLayer.output);
}

template<bool IsTemporal>
void Denoiser::PrepareAOVLayers_(const AOVInfo &info)
{
    const auto &imageInfos = info.imageInfos;
    for (std::size_t i = 1; i < imageInfos.size(); i++)
    {
        auto &newLayer = layers_.emplace_back();
        newLayer.input = imageInfos[i].input;
        newLayer.output = imageInfos[i].output;

        UnifyImageSize(newLayer, info.width, info.height);

        TryAllocOnDevice_(newLayer.input);
        TryAllocOnDevice_(newLayer.output);

        if constexpr (IsTemporal)
        {
            newLayer.previousOutput = newLayer.output;
        }
    }
}

/// @brief Set flow and flowtrust to `guideLayerTemporalBuffer_` if they're not
/// on device buffers. Particularly, guideLayerTemporalBuffer_ will also be
/// allocated with the additional size of two internal layers and set
/// `guideLayerTemporalBufferSize_`.
/// @param flowInfo
/// @return Return the size of internal layer.
std::size_t Denoiser::SetTemporalGuideInputs_(const FlowInfo &flowInfo)
{
    auto &newLayer = layers_.back();
    unsigned int width = newLayer.input.width, height = newLayer.input.height;

    std::size_t flowSize =
                    SetFlow(guideLayer_.flow, flowInfo.flow, width, height),
                flowTrustSize = 0,
                internalSize = GetOutputInternalSize(newLayer);

    if (const auto &flowTrust = flowInfo.flowTrust; flowTrust.has_value())
    {
        flowTrustSize = SetFlowTrust(guideLayer_.flowTrustworthiness,
                                     *flowTrust, width, height);
    }

    // Unknown, conservative one; we should ensure flowAlignment also satisfies
    // internal alignment.
    static constexpr std::size_t flowAlignment = 128;
    std::size_t flowAlignedSize =
                    UniUtils::RoundUpNonNegative(flowSize, flowAlignment),
                flowTrustAlignedSize =
                    UniUtils::RoundUpNonNegative(flowTrustSize, flowAlignment);
    guideLayerTemporalBufferSize_ =
        flowAlignedSize + flowTrustAlignedSize + internalSize + internalSize;
    guideLayerTemporalBuffer_ =
        HostUtils::DeviceMakeUninitializedUnique<std::byte[]>(
            guideLayerTemporalBufferSize_);

    CopyOrZero(guideLayerTemporalBuffer_.get(), 0, flowSize, guideLayer_.flow);
    CopyOrZero(guideLayerTemporalBuffer_.get(), flowAlignedSize, flowTrustSize,
               guideLayer_.flowTrustworthiness);

    return internalSize;
}

/// @brief Set the internal layers allocated by `SetTemporalGuideInputs_`.
/// Previous layer will be memset to 0.
/// @param internalSize
void Denoiser::SetTemporalGuideOutputs_(std::size_t internalSize)
{
    auto &newLayer = layers_.back();
    auto previousOffset = guideLayerTemporalBufferSize_ - internalSize;
    SetOutputInternalLayer(guideLayer_.outputInternalGuideLayer,
                           guideLayerTemporalBuffer_.get() + previousOffset -
                               internalSize,
                           newLayer.output.width, newLayer.output.height);

    auto tPtr = guideLayerTemporalBuffer_.get() + previousOffset;
    auto dPtr = HostUtils::ToDriverPointer(tPtr);

    guideLayer_.previousOutputInternalGuideLayer =
        guideLayer_.outputInternalGuideLayer;
    // Previous buffer should be set to all zero.
    guideLayer_.previousOutputInternalGuideLayer.data = dPtr;
    cudaMemset(tPtr.get(), 0, internalSize);
}

void Denoiser::PrepareTemporalColorLayer_(const ImageIOPair &pair,
                                          const FlowInfo &flowInfo)
{
    PrepareColorLayer_(pair);

    // Optix doc: previousOutput is read in optixDenoiserInvoke before writing a
    // new output, so previousOutput could be set to output (the same buffer)
    // for efficiency if useful in the application.
    auto &newLayer = layers_.back();
    newLayer.previousOutput = newLayer.output;

    auto internalSize = SetTemporalGuideInputs_(flowInfo);
    SetTemporalGuideOutputs_(internalSize);
}

void Denoiser::PrepareTemporalColorLayer_(const ImageIOPair &pair,
                                          const FlowInfo &flowInfo,
                                          unsigned int width,
                                          unsigned int height)
{
    PrepareColorLayer_(pair, width, height);

    // Optix doc: previousOutput is read in optixDenoiserInvoke before writing a
    // new output, so previousOutput could be set to output (the same buffer)
    // for efficiency if useful in the application.
    auto &newLayer = layers_.back();
    newLayer.previousOutput = newLayer.output;

    auto internalSize = SetTemporalGuideInputs_(flowInfo);
    SetTemporalGuideOutputs_(internalSize);
}

void Denoiser::Denoise(float blendFactor, unsigned int offsetX,
                       unsigned int offsetY)
{
    assert(!layers_.empty());
    auto bufferPtr = assistBuffer_.get();
    OptixDenoiserParams params{};
    auto scratchBufferDPtr = GetScratchBufferPtr_();
    if (NeedIntensity())
    {
        assert(!HostUtils::TestEnum(mode_, Mode::LDR));
        auto dPtr = HostUtils::ToDriverPointer(bufferPtr);
        HostUtils::CheckOptixError(optixDenoiserComputeIntensity(
            denoiser_.get(), LocalContextSetter::GetCurrentCUDAStream(),
            &layers_[0].input, dPtr, scratchBufferDPtr, scratchBufferSize_));
        params.hdrIntensity = dPtr;
        bufferPtr = bufferPtr + 1;
    }
    if (NeedAverageColor())
    {
        assert(!HostUtils::TestEnum(mode_, Mode::LDR));
        auto dPtr = HostUtils::ToDriverPointer(bufferPtr);
        HostUtils::CheckOptixError(optixDenoiserComputeAverageColor(
            denoiser_.get(), LocalContextSetter::GetCurrentCUDAStream(),
            &layers_[0].input, dPtr, scratchBufferDPtr, scratchBufferSize_));
        params.hdrAverageColor = dPtr;
    }
    params.temporalModeUsePreviousLayers =
        HostUtils::TestEnum(mode_, Mode::TemporalNotFirstFrame);
    params.blendFactor = blendFactor;

    HostUtils::CheckOptixError(optixDenoiserInvoke(
        denoiser_.get(), LocalContextSetter::GetCurrentCUDAStream(), &params,
        GetStateBufferPtr_(), stateBufferSize_, &guideLayer_, layers_.data(),
        static_cast<unsigned int>(layers_.size()), offsetX, offsetY,
        scratchBufferDPtr, scratchBufferSize_));
}

template<OptixDenoiserModelKind Kind>
BasicDenoiser<Kind>::BasicDenoiser(const BasicInfo &basicInfo,
                                   const ColorInfo &colorInfo,
                                   const TileInfo *tileInfo,
                                   const GuideInfo *guideInfo)
    : Denoiser{ Kind, basicInfo, guideInfo }
{
    colorInfo.CheckValid();
    if constexpr (Kind == OPTIX_DENOISER_MODEL_KIND_LDR)
        mode_ |= Mode::LDR | Mode::NoIntensity | Mode::NoAverageColor;

    auto width = colorInfo.GetMaxWidth(), height = colorInfo.GetMaxHeight();
    SetupDenoiser_(tileInfo, width, height);
    PrepareColorLayer_(colorInfo.imageInfo);
    TrySetOtherGuides_(guideInfo, width, height);
}

template class BasicDenoiser<OPTIX_DENOISER_MODEL_KIND_LDR>;
template class BasicDenoiser<OPTIX_DENOISER_MODEL_KIND_HDR>;

TemporalDenoiser::TemporalDenoiser(const BasicInfo &basicInfo,
                                   const ColorInfo &colorInfo,
                                   const FlowInfo &flowInfo,
                                   const TileInfo *tileInfo,
                                   const GuideInfo *guideInfo)
    : Denoiser{ OPTIX_DENOISER_MODEL_KIND_TEMPORAL, basicInfo, guideInfo }
{
    colorInfo.CheckValid();
    mode_ |= Mode::Temporal;
    auto width = colorInfo.GetMaxWidth(), height = colorInfo.GetMaxHeight();
    SetupDenoiser_(tileInfo, width, height);
    PrepareTemporalColorLayer_(colorInfo.imageInfo, flowInfo);
    TrySetOtherGuides_(guideInfo, width, height);
}

template<OptixDenoiserModelKind Kind>
AOVBasicDenoiser<Kind>::AOVBasicDenoiser(const BasicInfo &basicInfo,
                                         const AOVInfo &aovInfo,
                                         const TileInfo *tileInfo,
                                         const GuideInfo *guideInfo)
    : Denoiser{ Kind, basicInfo, guideInfo }
{
    aovInfo.CheckValid();
    auto width = aovInfo.GetMaxWidth(), height = aovInfo.GetMaxHeight();
    SetupDenoiser_(tileInfo, width, height);
    PrepareColorLayer_(aovInfo.GetColorInfo(), aovInfo.width, aovInfo.height);
    PrepareAOVLayers_<false>(aovInfo);
    TrySetOtherGuides_(guideInfo, width, height);
}

template class AOVBasicDenoiser<OPTIX_DENOISER_MODEL_KIND_AOV>;
template class AOVBasicDenoiser<OPTIX_DENOISER_MODEL_KIND_UPSCALE2X>;

template<OptixDenoiserModelKind Kind>
AOVTemporalBasicDenoiser<Kind>::AOVTemporalBasicDenoiser(
    const BasicInfo &basicInfo, const AOVInfo &aovInfo,
    const FlowInfo &flowInfo, const TileInfo *tileInfo,
    const GuideInfo *guideInfo)
    : Denoiser{ Kind, basicInfo, guideInfo }
{
    aovInfo.CheckValid();
    mode_ |= Mode::Temporal;
    auto width = aovInfo.GetMaxWidth(), height = aovInfo.GetMaxHeight();
    SetupDenoiser_(tileInfo, width, height);
    PrepareTemporalColorLayer_(aovInfo.GetColorInfo(), flowInfo, aovInfo.width,
                               aovInfo.height);
    PrepareAOVLayers_<true>(aovInfo);
    TrySetOtherGuides_(guideInfo, width, height);
}

template class AOVTemporalBasicDenoiser<OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV>;
template class AOVTemporalBasicDenoiser<
    OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X>;

} // namespace EasyRender::Optix
