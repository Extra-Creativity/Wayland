#include "Core/DeviceManager.h"
#include "Utils/Common.h"
#include "Utils/PrintUtils.h"
#include "HostUtils/DeviceAllocators.h"

using namespace EasyRender::Optix;

namespace EasyRender
{

DeviceManager::DeviceManager()
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
    return;
}

void DeviceManager::SetupOptix(SceneManager &scene, MainWindow &window,
                               std::string_view programSrc,
                               ProgramManager *program)
{
    BuildAccelStructure(scene);
    BuildPipeline(programSrc);
    BuildSBT(program);

    /* Temporary */
    int wSize = window.size.w * window.size.h * 4;
    cudaMalloc((void**) & deviceFrameBuffer, wSize);
    return;
}

void DeviceManager::BuildAccelStructure(SceneManager &scene)
{
    asBuildInput = make_unique<TriangleBuildInputArray>(scene.meshes.size());
    for (int i = 0; i < scene.meshes.size(); ++i)
    {
        asBuildInput->AddBuildInput(
            { VecToSpan<const float, 3>(scene.meshes[i]->vertex) },
            VecToSpan<int, 3>(scene.meshes[i]->index),
            GeometryFlags::DiableAnyHit);
    }
    as = make_unique<StaticAccelStructure>(*asBuildInput);
    cudaDeviceSynchronize();
    return;
}

void DeviceManager::BuildPipeline(string_view programSrc)
{
    assert(as);
    auto pipelineOption = PipelineConfig::GetDefault().SetNumPayloadValues(2);
    module = make_unique<Module>(programSrc, ModuleConfig::GetDefaultRef(),
                                 pipelineOption);
    pg = make_unique<ProgramGroupArray>();
    module->IdentifyPrograms({ string(programSrc) }, *pg);
    pipeline =
        make_unique<Pipeline>(*pg, 1, as->GetDepth(), 0, 0, pipelineOption);
}

void DeviceManager::BuildSBT(ProgramManager *program)
{
    assert(pg);
    sbt = make_unique<Optix::ShaderBindingTable>(program->GenerateSBT(*pg));
}

void DeviceManager::Launch(ProgramManager *program, WinSize wSize)
{
    assert(as);
    assert(pipeline);
    assert(sbt);

    Launcher launcher{ program->GetParamPtr(), program->GetParamSize() };
    launcher.Launch(*pipeline, LocalContextSetter::GetCurrentCUDAStream(), *sbt,
                    wSize.w, wSize.h);
    cudaDeviceSynchronize();
    return;
}

void DeviceManager::DownloadFrameBuffer(MainWindow &window) const
{
    auto size = window.size.h * window.size.w * 4;
    cudaMemcpy( window.frameBuffer.data(), deviceFrameBuffer, size,
               cudaMemcpyDeviceToHost);
}

OptixTraversableHandle DeviceManager::GetTraversableHandle() {
    assert(as);
    return as->GetHandle();
}

} // namespace EasyRender