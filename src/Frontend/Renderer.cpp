#include "Renderer.h"

namespace Wayland
{
Renderer::Renderer(WinSize s, std::string sceneSrc) : window(s), scene(sceneSrc)
{
    SetOptixEnvironment();
    BuildOptixAccelStructure();
}

void Renderer::Run()
{
    while (!window.ShouldClose())
    {
        window.Update();
    }
}

void Renderer::SetOptixEnvironment()
{
    optixContextManager.SetCachePath(".");
    Optix::Module::SetOptixSDKPath(OPTIX_DIR);
    Optix::Module::AddArgs("-I\"" GLM_DIR "\"");
    Optix::Module::AddArgs("-I\"" UTILS_INC_PATH "\"");
    Optix::Module::AddArgs("-diag-suppress 20012 -diag-suppress 3012");
    return;
}

void Renderer::BuildOptixAccelStructure()
{
    Optix::TriangleBuildInputArray buildInputs{ scene.meshes.size() };

    /*for (int i = 0; i < scene.meshes.size(); ++i)
    {
		buildInputs.AddBuildInput({ scene.meshes[i].vertices },
            								  scene.meshes[i].triangles, scene.meshes[i].inputFlags,
            								  scene.meshes[i].mat_indices);
	}
    StaticAccelStructure as{ buildInputs };*/
	cudaDeviceSynchronize();
    return;
}

} // namespace Wayland
