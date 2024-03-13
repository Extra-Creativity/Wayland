#pragma once
#include <memory>
#include "MainWindow.h"

namespace Wayland
{

class Renderer
{
public:
    Renderer(WinSize s);
    ~Renderer() {}
    void Run();

public:
    MainWindow window;
};

} // namespace Wayland