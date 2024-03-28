#include "Core/DeviceManager.h"
#include "Device/Light.h"
#include "HostUtils/DeviceAllocators.h"
#include "Utils/Common.h"
#include "Utils/PrintUtils.h"
#include "spdlog/spdlog.h"

using namespace EasyRender::Optix;
using namespace std;

namespace EasyRender
{

DeviceManager::DeviceManager()
    : frameBufferSize(0), indexBufferSize(0), normalBufferSize(0)
{
    SetEnvironment();
}

void DeviceManager::SetEnvironment()
{
    contextManager.SetCachePath(".");
    Module::SetOptixSDKPath(OPTIX_DIR);
    Module::AddArgs("-I\"" GLM_DIR "\"");
    Module::AddArgs("-I\"" UTILS_INC_PATH "\"");
    Module::AddArgs("-I\"" FRONT_PATH "\"");
    Module::AddArgs("-diag-suppress 20012 -diag-suppress 3012");
}

void DeviceManager::SetupOptix(SceneManager &scene, MainWindow &window,
                               std::string_view programSrc,
                               ProgramManager *program)
{
    AllocDeviceBuffer(scene, window.size);
    BuildAccelStructure(scene);
    BuildPipeline(programSrc);
    BuildSBT(program);
}

void DeviceManager::BuildAccelStructure(SceneManager &scene)
{
    asBuildInput =
        std::make_unique<TriangleBuildInputArray>(scene.meshes.size());
    for (int i = 0; i < scene.meshes.size(); ++i)
    {
        asBuildInput->AddBuildInput(
            { VecToSpan<const float, 3>(scene.meshes[i]->vertex) },
            VecToSpan<int, 3>(scene.meshes[i]->triangle),
            GeometryFlags::DiableAnyHit);
    }
    as = std::make_unique<StaticAccelStructure>(*asBuildInput);
    cudaDeviceSynchronize();
}

void DeviceManager::BuildPipeline(std::string_view programSrc)
{
    assert(as);
    auto pipelineOption = PipelineConfig::GetDefault().SetNumPayloadValues(2);
    module = std::make_unique<Module>(programSrc, ModuleConfig::GetDefaultRef(),
                                      pipelineOption);
    pg = std::make_unique<ProgramGroupArray>();
    module->IdentifyPrograms({ string(programSrc) }, *pg);
    pipeline = std::make_unique<Pipeline>(*pg, 1, as->GetDepth(), 0, 0,
                                          pipelineOption);
}

void DeviceManager::BuildSBT(ProgramManager *program)
{
    assert(pg);
    assert(frameBufferSize > 0);
    sbt =
        std::make_unique<Optix::ShaderBindingTable>(program->GenerateSBT(*pg));
}

void DeviceManager::AllocDeviceBuffer(SceneManager &scene, glm::ivec2 wSize)
{
    frameBufferSize = wSize.x * wSize.y * sizeof(glm::u8vec4);
    cudaMalloc(&d_FrameBuffer, frameBufferSize);
    spdlog::info("Malloc {} bytes at device -> frameBuffer", frameBufferSize);

    assert(scene.triangleNum % 3 == 0);
    indexBufferSize = scene.triangleNum * sizeof(glm::ivec3);
    cudaMalloc(&d_IndexBuffer, indexBufferSize);
    spdlog::info("Malloc {} bytes at device -> indexBuffer", indexBufferSize);

    normalBufferSize = scene.vertexNum * sizeof(glm::vec3);
    cudaMalloc(&d_NormalBuffer, normalBufferSize);
    spdlog::info("Malloc {} bytes at device -> vertexBuffer", normalBufferSize);

    /* We only have areaLight, for now */
    areaLightBufferSize = scene.lights.size() * sizeof(Device::DeviceAreaLight);
    areaLightBufferSize += scene.areaLightVertexNum * sizeof(glm::vec3);
    cudaMalloc(&d_AreaLightBuffer, areaLightBufferSize);
    d_AreaLightVertexBuffer =
        reinterpret_cast<glm::vec3 *>(d_AreaLightBuffer + scene.lights.size());
    spdlog::info("Malloc {} bytes at device -> areaLightBuffer",
                 areaLightBufferSize);
}

void DeviceManager::Launch(ProgramManager *program, glm::ivec2 wSize)
{
    assert(as);
    assert(pipeline);
    assert(sbt);

    Launcher launcher{ program->GetParamPtr(), program->GetParamSize() };
    launcher.Launch(*pipeline, LocalContextSetter::GetCurrentCUDAStream(), *sbt,
                    wSize.x, wSize.y);
    cudaDeviceSynchronize();
}

void DeviceManager::DownloadFrameBuffer(MainWindow &window) const
{
    int size = window.size.x * window.size.y * 4;
    cudaMemcpy(window.frameBuffer.data(), d_FrameBuffer, size,
               cudaMemcpyDeviceToHost);
}

OptixTraversableHandle DeviceManager::GetTraversableHandle()
{
    assert(as);
    return as->GetHandle();
}

} // namespace EasyRender