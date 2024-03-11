#include "Optix/Core/Optix-All.h"

#include "Camera/Camera.h"
#include "Common.h"
#include "SimpleModel/Model.h"

#include "Param.h"

using namespace Wayland;

void FillColors(const aiMaterial *material, Example::Phong::HitData &data)
{
    aiColor3D color;
    material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
    data.kd = { color.r, color.g, color.b };

    material->Get(AI_MATKEY_COLOR_SPECULAR, color);
    data.ks = { color.r, color.g, color.b };
    if (data.ks == glm::zero<glm::vec3>()) // add a slight specular effect.
        data.ks = glm::vec3{ 0.1f };

    material->Get(AI_MATKEY_COLOR_AMBIENT, color);
    data.ka = { color.r, color.g, color.b };
    data.sPow = 32;
}

auto GenerateHitData(const Example::StaticModel &model,
                     unsigned int buildInputID, unsigned int sbtID,
                     [[maybe_unused]] unsigned int)
    -> std::pair<Example::Phong::HitData, std::size_t>
{
    const aiMesh *mesh = model.GetScene()->mMeshes[buildInputID];

    aiString path;
    auto currMat = model.GetScene()->mMaterials[mesh->mMaterialIndex];
    currMat->GetTexture(aiTextureType_DIFFUSE, 0, &path);
    auto texture = model.GetTexture(path);
    assert(texture);

    Example::Phong::HitData data{
        .diffuseTexture = texture->GetHandle(),
        .indices = model.GetIndices(buildInputID),
        .texCoords = model.GetDeviceTextureBuffer(buildInputID),
        .normals = model.GetDeviceNormalBuffer(buildInputID)
    };
    FillColors(currMat, data);

    return std::pair{ data, 2 };
}

Optix::ShaderBindingTable GetSBT(const Example::StaticModel &model,
                                 const Optix::ProgramGroupArray &arr)
{
    Optix::SBTData<void> raygen; // empty
    Optix::SBTData<Example::Phong::MissData> miss{
        .data = { .bgColor = { 1.0f, 1.0f, 1.0f } }
    };
    auto info =
        Optix::GetSBTHitRecordBuffer<Example::Phong::HitData>(1, model.GetAS());
    return Optix::ShaderBindingTable{
        raygen, 0, miss, 1, info.hitRecords, info.groupIndices.data(), arr
    };
}

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
            Optix::PipelineConfig::GetDefault().SetNumPayloadValues(3)
        };
        Optix::ProgramGroupArray arr{ 3 };
        myModule.IdentifyPrograms({ SHADER_PATH }, arr);
        Optix::Pipeline pipeline{
            arr, 1, model.GetAS().GetDepth(),
            0,   0, Optix::PipelineConfig::GetDefault().SetNumPayloadValues(3)
        };

        Optix::ShaderBindingTable sbt = GetSBT(model, arr);

        unsigned int width = 1024, height = 1024;
        auto buffer = HostUtils::DeviceMakeUninitializedUnique<glm::u8vec4[]>(
            width * height);
        auto rawBuffer = buffer.get().get();
        Example::Camera camera{ { 0, 10, 35 }, { 0, 1, 0 }, { 0, 0, -1 } };
        Optix::Launcher launcher{ Example::Phong::LaunchParam{
            .lightPos = { -3, 12, 7.5f },
            .lightColor = { 25.0f, 25.0f, 25.0f },
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