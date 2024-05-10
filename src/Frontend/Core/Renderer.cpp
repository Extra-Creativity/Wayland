#include "Core/Renderer.h"
#include "Programs/Programs-All.h"

 #define STB_IMAGE_WRITE_IMPLEMENTATION
 #include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <chrono>
#include <ctime>

using namespace std;
using namespace chrono;

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

    high_resolution_clock::time_point TP1 = high_resolution_clock::now();
    duration<size_t, std::nano> dur;
    int frameCnt = 0;

    while (!window.ShouldClose())
    {
        device.Launch(program.get(), window.size);
        device.DownloadFrameBuffer(window);
        window.Update();
        program->Update();

        ++frameCnt;
        high_resolution_clock::time_point TP2 = high_resolution_clock::now();
        dur = TP2 - TP1;
        bool outTime =
            setting.timeLimit > 0 &&
            setting.timeLimit * 1000 < duration_cast<milliseconds>(dur).count();
        bool outFrame =
            setting.frameLimit > 0 && frameCnt >= setting.frameLimit;
        if (outTime || outFrame)
            break;
    }
    if (setting.writeOutput)
        WriteOutput(float(duration_cast<microseconds>(dur).count()) / 1000.f /
                        1000.f, frameCnt);
    program->End();
}

/* RenderTime in seconds */
void Renderer::WriteOutput(float renderTime, uint32_t frameCnt)
{
    time_t now = time(0);
    string curStr = ctime(&now);
    curStr = curStr.substr(0, curStr.find("\n"));
    curStr.replace(curStr.find(":"), 1, "_");
    curStr.replace(curStr.find(":"), 1, "_");
    curStr = string("[") + curStr + string("]");

    string renderTimeStr = to_string(renderTime);
    renderTimeStr = renderTimeStr.substr(0, renderTimeStr.find(".") + 2) + "s";

    string fileStr = setting.outputPath + "/" + setting.sceneName;
    fileStr += "-" + PROGRAM_NAME[(int)programType];
    fileStr += "-" + renderTimeStr + "-" + curStr;

    string txtStr = fileStr + ".txt";

    ostringstream oss;
    ofstream ofs;
    ofs.open(txtStr);
    ofs << "{\n";
    ofs << "\tScene: " << setting.sceneName << "\n";
    ofs << "\tProgram: " << PROGRAM_NAME[(int)programType] << "\n";
    ofs << "\tResolution: " << setting.resolution.x << " "
        << setting.resolution.y << "\n";
    ofs << "\tRenderTime: " << renderTime << "\n";
    ofs << "\tFrameCount: " << frameCnt << "\n";
    ofs << "\tFrameTime: " << renderTime / float(frameCnt) << "\n";
    ofs << "\tFramesPerSecond: " << float(frameCnt) / renderTime << "\n";
    ofs << "}\n";
    ofs.close();

    string imgStr = fileStr + ".png";

    vector<glm::u8vec4> imgBuffer = window.frameBuffer;
    reverse(imgBuffer.begin(), imgBuffer.end());
    for (int i = 0; i < window.size.y; ++i)
    {
        auto begin = imgBuffer.begin() + i * window.size.x;
        reverse(begin, begin + window.size.x);
    }

    stbi_write_png(imgStr.c_str(), window.size.x, window.size.y, 4,
                   imgBuffer.data(), window.size.x * 4);
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
			timeLimit = static_cast<float>(atof(argv[++i]));
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
    writeOutput = true;
}

void RendererSetting::SetDefault()
{
    frameLimit = -1;
    timeLimit = -1.f;
    scenePath = R"(..\..\..\scene\cornell-box\cbox.pbrt)";
    SetSceneName();
    outputPath = R"(..\..\..)";
    program = ProgramType::PathTracing;
    resolution = glm::ivec2(1000, 1000);
    writeOutput = false;

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

void RendererSetting::SetFrameLimit(int f)
{
    frameLimit = f;
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
    writeOutput = true;
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
