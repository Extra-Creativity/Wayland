#include "MainWindow.h"
#include "HostUtils/ErrorCheck.h"
#include <assert.h>

using namespace Wayland;

MainWindow::MainWindow(WinSize s)
{
    setSize(s);
    Init();
    putInCenter();
}

void MainWindow::Init()
{
    GLFWwindow *window;

    /* Initialize the library */
    int glfwInitResult = glfwInit();
    HostUtils::CheckError(glfwInitResult, "Fail to initialize glfw.");
    if (!glfwInitResult)
        return;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(size.w, size.h, "Wayland", NULL, NULL);
    HostUtils::CheckError(window != nullptr, "Fail to create glfw window.");
    if (!window)
    {
        glfwTerminate();
        return;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glfWindow = window;
}

void MainWindow::Update()
{
    /* Render here */
    glClear(GL_COLOR_BUFFER_BIT);

    /* Swap front and back buffers */
    glfwSwapBuffers(glfWindow);

    /* Poll for and process events */
    glfwPollEvents();
}

/* Put the window in the center of user's monitor*/
void MainWindow::putInCenter()
{
    assert(glfWindow);

    // Defining a monitor
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *mode = glfwGetVideoMode(monitor);

    // Putting it in the centre
    int xpos = max(0, (mode->width - size.w) / 2);
    int ypos = max(0, (mode->height - size.h) / 2);
    glfwSetWindowPos(glfWindow, xpos, ypos);
}

bool MainWindow::ShouldClose()
{
    return glfwWindowShouldClose(glfWindow);
}
