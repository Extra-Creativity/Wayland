#include "Core/Renderer.h"
#include "Programs/Programs-All.h"

using namespace std;

namespace EasyRender
{

Renderer::Renderer(glm::ivec2 s, std::string_view sceneSrc,
                   std::string_view programSrc)
    : window(s), scene(sceneSrc), device()
{
     program =  make_unique<ColorProgramManager>(this);
    //program = make_unique<DepthProgramManager>(this);
    //program = make_unique<MeshProgramManager>(this);
     //program = make_unique<SimpleProgramManager>(this);

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

} // namespace EasyRender
