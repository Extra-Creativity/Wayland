#include "Renderer.h"

using namespace Wayland;

Renderer::Renderer(WinSize s) : window(s) {}

void Renderer::Run()
{
    while (!window.ShouldClose())
    {
        window.Update();
    }
}

// namespace Wayland