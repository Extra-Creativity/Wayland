#include "Core/Renderer.h"
#include "Programs/Programs-All.h"

namespace Wayland
{

Renderer::Renderer(WinSize s, std::string_view sceneSrc,
                   std::string_view programSrc)
    : window(s), scene(sceneSrc), device()
{
    //program = make_unique<SimpleProgramManager>(this);
    program = make_unique<DepthProgramManager>(this);
    //program = make_unique<MeshProgramManager>(this);

    device.SetupOptix(scene, window, programSrc, program.get());
}


void Renderer::Run()
{
    program->Setup();
    while (!window.ShouldClose())
    {
        device.Launch(program.get(), window.size);
        device.DownloadFrameBuffer(window);
        window.Update();
        program->Update();
    }
}

} // namespace Wayland
