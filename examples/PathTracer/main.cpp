#include "Optix/Core/Optix-All.h"

#include "Camera/Camera.h"
#include "Common.h"
#include "SimpleModel/Model.h"

#include "Conversion.h"
#include "Param.h"

using namespace Wayland;

auto GenerateHitData(const Example::StaticModel &model,
                     unsigned int buildInputID, unsigned int sbtID,
                     [[maybe_unused]] unsigned int)
    -> std::pair<Example::PathTracing::HitData, std::size_t>
{
    const aiMesh *mesh = model.GetScene()->mMeshes[buildInputID];

    Example::PathTracing::HitData data{
        .indices = model.GetIndices(buildInputID),
        .normals = model.GetDeviceNormalBuffer(buildInputID)
    };

    auto material = model.GetScene()->mMaterials[mesh->mMaterialIndex];
    aiColor3D color;
    material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
    data.color = { color.r, color.g, color.b };

    material->Get(AI_MATKEY_COLOR_EMISSIVE, color);
    data.emission = { color.r, color.g, color.b };

    return std::pair{ data, 2 };
}

Optix::ShaderBindingTable GetSBT(const Example::StaticModel &model,
                                 const Optix::ProgramGroupArray &arr)
{
    Optix::SBTData<void> raygen; // empty
    Optix::SBTData<void> miss;   // empty too.
    auto info = Optix::GetSBTHitRecordBuffer<Example::PathTracing::HitData>(
        1, model.GetAS());
    return Optix::ShaderBindingTable{
        raygen, 0, miss, 1, info.hitRecords, info.groupIndices.data(), arr
    };
}

const int maxDepth = 10;
const bool needDenoise = true;

HostUtils::DeviceUniquePtr<glm::vec4[]> Denoise(glm::vec4 *noisyBuffer,
                                                unsigned int width,
                                                unsigned int height)
{
    Optix::ColorInfo colorInfo{};
    std::size_t totalSize = std::size_t{ width } * height;
    // Denoiser only accepts fp16 or fp32; normally we can pass a float
    // buffer directly, but here we don't want to change shader, so it's
    // still u8vec4 and a vec4 buffer is allocated aside.
    auto denoisedBuffer =
        HostUtils::DeviceMakeUninitializedUnique<glm::vec4[]>(totalSize);

    Optix::SetTightOptixImage2D<Optix::PixelFormat::Float4>(
        colorInfo.imageInfo.input, noisyBuffer, width, height);
    colorInfo.imageInfo.output = colorInfo.imageInfo.input;
    colorInfo.imageInfo.output.data =
        HostUtils::ToDriverPointer(denoisedBuffer.get());

    Optix::LDRDenoiser denoiser{ Optix::BasicInfo{}, colorInfo };
    denoiser.Denoise();
    return denoisedBuffer;
}

int main()
{
    try
    {
        auto manager = Example::SetupEnvironment();
        auto model = Example::StaticModel{ MODEL_PATH, &GenerateHitData,
                                           Optix::GeometryFlags::DiableAnyHit };
        cudaDeviceSynchronize(); // For AS.

        auto pipelineConfig =
            Optix::PipelineConfig::GetDefault().SetNumPayloadValues(5);
        Optix::Module myModule{ SHADER_PATH,
                                Optix::ModuleConfig::GetDefaultRef(),
                                pipelineConfig };
        Optix::ProgramGroupArray arr{ 3 };
        myModule.IdentifyPrograms({ SHADER_PATH }, arr);
        Optix::Pipeline pipeline{ arr, maxDepth, model.GetAS().GetDepth(),
                                  2,   2,        pipelineConfig };

        Optix::ShaderBindingTable sbt = GetSBT(model, arr);

        unsigned int width = 1024, height = 1024;
        std::size_t bufferSize = width * height;
        auto buffer =
            HostUtils::DeviceMakeUninitializedUnique<glm::vec4[]>(bufferSize);
        auto rawBuffer = buffer.get().get();
        Example::Camera camera{ { -0.23, 2.58, 5 }, { 0, 1, 0 }, { 0, 0, -1 } };
        Optix::Launcher launcher{ Example::PathTracing::LaunchParam{
            .sampleNum = 1024,
            .maxDepth = maxDepth - 1,
            .camera = camera.ToDeviceCamera(1), // aspect is 1
            .colorBuffer = rawBuffer,
            .traversable = model.GetAS().GetHandle(),
        } };
        launcher.Launch(pipeline, 0, sbt, width, height);
        cudaDeviceSynchronize(); // For ray gen.

        if (needDenoise)
        {
            auto result = Denoise(rawBuffer, width, height);
            auto byteResult = FromFloatToChar(
                reinterpret_cast<float *>(result.get().get()), bufferSize * 4);
            Example::SaveImageCPU(IMAGE_PATH "-denoised", width, height,
                                  byteResult.data());
        }

        auto byteResult = FromFloatToChar(reinterpret_cast<float *>(rawBuffer),
                                          bufferSize * 4);
        Example::SaveImageCPU(IMAGE_PATH, width, height, byteResult.data());
    }
    catch (const std::exception &exception)
    {
        SPDLOG_ERROR("{}", exception.what());
        return 1;
    }
    return 0;
}