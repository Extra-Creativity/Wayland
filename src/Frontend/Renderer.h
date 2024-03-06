#pragma once
#include <memory>
#include "MainWindow.h"

namespace Wayland
{

class Renderer
{
public:
    Renderer(){};

    void Init(WinSize wSize);
    void Run();
    void Destroy();

public:
    MainWindowPtr window;
};

} // namespace Wayland