 #pragma once
#include <memory>
#include <string>
#include <unordered_map>

#include "Core/Optix-All.h"
#include "SceneManager.h"
#include "MainWindow.h"
#include "DeviceManager.h"
#include "ProgramManager.h"
#include "glm/glm.hpp"

namespace EasyRender
{

enum class ProgramType
{
    AO,
    Color,
    Depth,
    Mesh,
    Normal,
    Simple,
    Texture,
    WireFrame,
    PathTracing,
    RandomWalk,
    BDPT,
    ProgramTypeMax
};

const std::string PROGRAM_NAME[] = {
    "AO",          "Color",      "Depth",   "Mesh",
    "Normal",      "Simple",     "Texture", "WireFrame",
    "PathTracing", "RandomWalk", "BDPT",    "PROGRAM OUT OF INDEX"
};


/* TBD: use more robust path string */
const std::string PROGRAM_SRC[] = {
    R"(..\..\..\src\Frontend\Programs\AO\AO.cu)",
    R"(..\..\..\src\Frontend\Programs\TestExamples\Color\Color.cu)",
    R"(..\..\..\src\Frontend\Programs\TestExamples\Depth\Depth.cu)",
    R"(..\..\..\src\Frontend\Programs\TestExamples\Mesh\Mesh.cu)",
    R"(..\..\..\src\Frontend\Programs\TestExamples\Normal\Normal.cu)",
    R"(..\..\..\src\Frontend\Programs\TestExamples\Simple\Simple.cu)",
    R"(..\..\..\src\Frontend\Programs\TestExamples\Texture\Texture.cu)",
    R"(..\..\..\src\Frontend\Programs\TestExamples\WireFrame\WireFrame.cu)",
    R"(..\..\..\src\Frontend\Programs\PathTracing\PathTracing.cu)",
    R"(..\..\..\src\Frontend\Programs\RandomWalk\RandomWalk.cu)",
    R"(..\..\..\src\Frontend\Programs\BDPT\BDPT.cu)",
    R"(PROGRAM OUT OF INDEX)",
};

class RendererSetting
{
public:
    RendererSetting();
    RendererSetting(int argc, char **argv);
    ~RendererSetting() = default;
    void SetFrameLimit(int f);
    void SetTimeLimit(float time);
    void SetScenePath(std::string_view src);
    void SetOutputPath(std::string_view dst);
    void SetResolution(int x, int y);
    void SetProgram(ProgramType pg);

private:
    void SetSceneName();
    void SetDefault();

public:
    float timeLimit;
    int frameLimit;
    std::string scenePath;
    std::string outputPath;
    glm::ivec2 resolution;
    ProgramType program;
    std::string sceneName;
    bool writeOutput;

    std::unordered_map<std::string, ProgramType> programMap;
};

class Renderer
{
public:
    Renderer(RendererSetting &setting);
    ~Renderer() = default;
    void Run();

private:
    void SetProgram(ProgramType pgType);
    void WriteOutput(float renderTime, uint32_t frameCnt);

public:
    MainWindow window;
    SceneManager scene;
    /* Deal with GPU and OptiX */
    DeviceManager device;
    /* Polymorphic and refer to user defined ProgramManager */
    ProgramManagerPtr program;
    RendererSetting setting;

private:
    ProgramType programType;
    std::string programSrc;

};

} // namespace EasyRender