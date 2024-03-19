#include "Renderer.h"

using namespace Wayland;

Renderer::Renderer(WinSize s, std::string sceneSrc) : window(s), scene(sceneSrc) {}

void Renderer::Run()
{
    while (!window.ShouldClose())
    {
        window.Update();
    }
}

// namespace Wayland