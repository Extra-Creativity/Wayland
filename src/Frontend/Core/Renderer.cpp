#include "Core/Renderer.h"
#include "Programs/Programs-All.h"

#include <iostream>
#include <string>
#include <cstring>

using namespace std;

namespace EasyRender
{

Renderer::Renderer(RendererSetting &mySet)
    : window(mySet.resolution), scene(mySet.scenePath), device()
{
    SetProgram(mySet.program);
	device.SetupOptix(scene, window, programSrc, program.get());
    setting = mySet;
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
    assert((int)pgType >= 0 && (int)pgType < (int)ProgramType::ProgramTypeMax);
    programType = pgType;
    programSrc = PROGRAM_SRC[static_cast<int>(pgType)];
    switch (pgType) {
    case ProgramType::AO:
        program = make_unique<AOProgramManager>(this);
        break;
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
    case ProgramType::Texture:
		 program = make_unique<TextureProgramManager>(this);
		 break;
    case ProgramType::WireFrame:
         program = make_unique<WireFrameProgramManager>(this);
        break;
    case ProgramType::PathTracing:
        program = make_unique<PathTracingProgramManager>(this);
        break;
    case ProgramType::RandomWalk:
        program = make_unique<RandomWalkProgramManager>(this);
        break;
    case ProgramType::BDPT:
        program = make_unique<BDPTProgramManager>(this);
        break;
    default:
        assert(0);
    }
}

RendererSetting::RendererSetting()
{
	SetDefault();
}

RendererSetting::RendererSetting(int argc, char **argv)
{
    SetDefault();
    for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-t") == 0)
		{
			timeLimit = atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-s") == 0)
		{
			scenePath = argv[++i];
			SetSceneName();
		}
		else if (strcmp(argv[i], "-o") == 0)
		{
			outputPath = argv[++i];
		}
		else if (strcmp(argv[i], "-p") == 0)
		{
            if (programMap.find(argv[++i]) != programMap.end())
                program = programMap[argv[i]];
		}
		else if (strcmp(argv[i], "-r") == 0)
		{
			resolution.x = atoi(argv[++i]);
			resolution.y = atoi(argv[++i]);
		}
	}
}

void RendererSetting::SetDefault()
{
    timeLimit = -1.f;
    scenePath = R"(..\..\..\scene\cornell-box\cbox.pbrt)";
    SetSceneName();
    outputPath = R"(.)";
    program = ProgramType::PathTracing;
    resolution = glm::ivec2(1000, 1000);

    programMap.clear();
    for (int i = 0; i < (int)ProgramType::ProgramTypeMax; i++)
    {
        programMap[PROGRAM_NAME[i]] = (ProgramType)(i);
    }
}

void RendererSetting::SetSceneName()
{
    std::size_t pos1 = scenePath.rfind(R"(\)");
    assert(pos1 != std::string::npos);
    sceneName = scenePath.substr(pos1 + 1);
    std::size_t pos2 = sceneName.rfind(".");
    assert(pos2 != std::string::npos);
    sceneName = sceneName.substr(0, pos2);
}

void RendererSetting::SetTimeLimit(float time)
{
    timeLimit = time;
}

void RendererSetting::SetScenePath(std::string_view src)
{
    scenePath = src;
    SetSceneName();
}

void RendererSetting::SetOutputPath(std::string_view dst)
{
    outputPath = dst;
}

void RendererSetting::SetResolution(int x, int y)
{
    resolution = { x, y };
}

void RendererSetting::SetProgram(ProgramType pg)
{
    program = pg;
}
        
} // namespace EasyRender
