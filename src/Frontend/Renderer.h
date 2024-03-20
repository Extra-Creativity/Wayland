 #pragma once
#include <memory>
#include <string>
#include "MainWindow.h"
#include "SceneManager.h"
#include "Core/Optix-All.h"

namespace Wayland
{

class Renderer
{
public:
    Renderer(WinSize s, std::string sceneSrc);
    ~Renderer() = default;
    void Run();

private:
    void SetOptixEnvironment();
    void BuildOptixAccelStructure();

public:
    MainWindow window;
    SceneManager scene;

private:
    Optix::ContextManager optixContextManager;
    Optix::TriangleBuildInputArray buildInputs;

};

} // namespace Wayland