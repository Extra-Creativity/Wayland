 #pragma once
#include <memory>
#include <string>

#include "Core/Optix-All.h"
#include "Core/SceneManager.h"
#include "Core/MainWindow.h"
#include "Core/DeviceManager.h"
#include "Core/ProgramManager.h"


namespace Wayland
{

class Renderer
{
public:
    Renderer(WinSize s, std::string_view sceneSrc, std::string_view programSrc);
    //Renderer(WinSize s);
    ~Renderer() = default;
    void Run();

public:
    MainWindow window;
    SceneManager scene;
    /* Deal with GPU and OptiX */
    DeviceManager device;
    /* Polymorphic and refer to user defined ProgramManager */
    ProgramManagerPtr program;
};

} // namespace Wayland