#pragma once
#include <GLFW/glfw3.h>
#include <memory>
#include <utility>
#include <vector>
#include <glm/glm.hpp>

namespace EasyRender
{

class MainWindow
{
public:
    MainWindow(glm::ivec2 s);
    ~MainWindow() { glfwTerminate(); }

    void Update();
    void PutInCenter();
    bool ShouldClose();

    void SetSize(glm::ivec2 s)
    {
        size = s;
        frameBuffer.resize(size.x * size.y);
    }

private:
    void Init();
    void DisplayFrameBuffer();

public:
    glm::ivec2 size;
    std::vector<glm::u8vec4> frameBuffer;

private:
    GLFWwindow *glfWindow;
    GLuint fbTexture{ 0 };
};

using MainWindowPtr = std::unique_ptr<MainWindow>;

} // namespace EasyRender
