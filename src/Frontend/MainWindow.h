#pragma once
#include <GLFW/glfw3.h>
#include <memory>
#include <utility>

namespace Wayland
{
    struct WinSize
    {
	    int w;
	    int h;
    };

class MainWindow
{
public:
    MainWindow() : glfWindow(nullptr) {}

    void Init();
    void Update();
    void Destroy();
    void putInCenter();
    bool ShouldClose();

    void setSize(WinSize s) { size = s; }

private:
    GLFWwindow *glfWindow;
    WinSize size;
};

using MainWindowPtr = std::unique_ptr<MainWindow>;

} // namespace Wayland
