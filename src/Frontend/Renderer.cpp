#include "Renderer.h"

using namespace Wayland;

void Renderer::Init(WinSize wSize)
{
    window = std::make_unique<MainWindow>();
    window->setSize(wSize);
    window->Init();
    window->putInCenter();
}

void Renderer::Run()
{
    while (!window->ShouldClose())
    {
        window->Update();
    }
}

void Renderer::Destroy()
{
    window->Destroy();
}

// namespace Wayland