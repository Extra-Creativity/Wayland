#pragma once
#include <memory>
#include <string>
#include "MainWindow.h"
#include "SceneManager.h"

namespace Wayland
{

class Renderer
{
public:
    Renderer(WinSize s, std::string sceneSrc);
    ~Renderer() = default;
    void Run();

public:
    MainWindow window;
    SceneManager scene;

};

} // namespace Wayland