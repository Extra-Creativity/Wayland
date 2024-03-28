 #pragma once
#include <memory>
#include <string>

#include "Core/Optix-All.h"
#include "SceneManager.h"
#include "MainWindow.h"
#include "DeviceManager.h"
#include "ProgramManager.h"

namespace EasyRender
{

enum class ProgramType
{
    Color,
    Depth,
    Mesh,
    Normal,
    Simple,
    WireFrame,
    PathTracing,
    ProgramTypeMax
};

/* TBD: use more robust path string */
const std::string PROGRAM_SRC[] = {
    R"(..\..\..\src\Frontend\Programs\TestExamples\Color\Color.cu)",
    R"(..\..\..\src\Frontend\Programs\TestExamples\Depth\Depth.cu)",
    R"(..\..\..\src\Frontend\Programs\TestExamples\Mesh\Mesh.cu)",
    R"(..\..\..\src\Frontend\Programs\TestExamples\Normal\Normal.cu)",
    R"(..\..\..\src\Frontend\Programs\TestExamples\Simple\Simple.cu)",
    R"(..\..\..\src\Frontend\Programs\TestExamples\WireFrame\WireFrame.cu)",
    R"(..\..\..\src\Frontend\Programs\PathTracing\PathTracing.cu)",
    R"(PROGRAM OUT OF INDEX)",
};

class Renderer
{
public:
    Renderer(glm::ivec2 s, ProgramType pgType, std::string_view sceneSrc);
    //Renderer(glm::ivec2 s);
    ~Renderer() = default;
    void Run();

private:
    void SetProgram(ProgramType pgType);

public:
    MainWindow window;
    SceneManager scene;
    /* Deal with GPU and OptiX */
    DeviceManager device;
    /* Polymorphic and refer to user defined ProgramManager */
    ProgramManagerPtr program;

private:
    ProgramType programType;
    std::string programSrc;

};

} // namespace EasyRender