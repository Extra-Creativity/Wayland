#pragma once
#include <GLFW/glfw3.h>
#include <memory>
#include <utility>
#include <vector>
#include <glm/glm.hpp>

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
    MainWindow(WinSize s);
    ~MainWindow() { glfwTerminate(); }

    void Update();
    void PutInCenter();
    bool ShouldClose();

    void SetSize(WinSize s)
    {
        size = s;
        frameBuffer.resize(size.w * size.h);
    }

private:
    void Init();
    void DisplayFrameBuffer();

public:
    WinSize size;
    std::vector<glm::u8vec4> frameBuffer;

private:
    GLFWwindow *glfWindow;
    GLuint fbTexture{ 0 };
};

using MainWindowPtr = std::unique_ptr<MainWindow>;

} // namespace Wayland
