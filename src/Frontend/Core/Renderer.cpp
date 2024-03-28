#include "Core/Renderer.h"
#include "Programs/Programs-All.h"

using namespace std;

namespace EasyRender
{

Renderer::Renderer(glm::ivec2 s, ProgramType pgType, std::string_view sceneSrc)
    : window(s), scene(sceneSrc), device()
{
    SetProgram(pgType);
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
    program->End();
}

void Renderer::SetProgram(ProgramType pgType)
{
    assert(pgType > 0 && pgType < ProgramType::ProgramTypeMax);
    programType = pgType;
    programSrc = PROGRAM_SRC[static_cast<int>(pgType)];
    switch (pgType) {
    case ProgramType::Color:
         program = make_unique<ColorProgramManager>(this);
        break;
    case ProgramType::Depth:
         program = make_unique<DepthProgramManager>(this);
         break;
    case ProgramType::Mesh:
         program = make_unique<MeshProgramManager>(this);
         break;
    case ProgramType::Normal:
         program = make_unique<NormalProgramManager>(this);
         break;
    case ProgramType::Simple:
         program = make_unique<SimpleProgramManager>(this);
         break;
    case ProgramType::WireFrame:
         program = make_unique<WireFrameProgramManager>(this);
        break;
    case ProgramType::PathTracing:
        program = make_unique<PathTracingProgramManager>(this);
        break;
    default:
        assert(0);
    }
}

} // namespace EasyRender
