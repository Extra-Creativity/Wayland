#include "Optix/Core/Optix-All.h"

#include "Camera/Camera.h"
#include "Common.h"
#include "SimpleModel/Model.h"

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

int main()
{
    try
    {
        auto manager = Example::SetupEnvironment();
        auto model = Example::StaticModel{ MODEL_PATH, &GenerateHitData,
                                           Optix::GeometryFlags::DiableAnyHit };
        cudaDeviceSynchronize(); // For AS.
        Optix::Module myModule{
            SHADER_PATH, Optix::ModuleConfig::GetDefaultRef(),
            Optix::PipelineConfig::GetDefault().SetNumPayloadValues(5)
        };
        Optix::ProgramGroupArray arr{ 3 };
        myModule.IdentifyPrograms({ SHADER_PATH }, arr);
        Optix::Pipeline pipeline{
            arr,
            maxDepth,
            model.GetAS().GetDepth(),
            2,
            2,
            Optix::PipelineConfig::GetDefault().SetNumPayloadValues(5)
        };

        Optix::ShaderBindingTable sbt = GetSBT(model, arr);

        unsigned int width = 1024, height = 1024;
        auto buffer = HostUtils::DeviceMakeUninitializedUnique<glm::u8vec4[]>(
            width * height);
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

        Example::SaveImage(IMAGE_PATH, width, height, rawBuffer);
    }
    catch (const std::exception &exception)
    {
        SPDLOG_ERROR("{}", exception.what());
        return 1;
    }
    return 0;
}