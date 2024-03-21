#include "Core/MainWindow.h"
#include "HostUtils/ErrorCheck.h"
#include <assert.h>

using namespace Wayland;

MainWindow::MainWindow(WinSize s)
{
    SetSize(s);
    Init();
    PutInCenter();
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
    DisplayFrameBuffer();
    /* Swap front and back buffers */
    glfwSwapBuffers(glfWindow);
    /* Poll for and process events */
    glfwPollEvents();
}

/* Need to swap buffer after this function to actually see results */
void MainWindow::DisplayFrameBuffer()
{
    if (fbTexture == 0)
        glGenTextures(1, &fbTexture);

    glBindTexture(GL_TEXTURE_2D, fbTexture);
    GLenum texFormat = GL_RGBA;
    GLenum texelType = GL_UNSIGNED_BYTE;
    glTexImage2D(GL_TEXTURE_2D, 0, texFormat, size.w, size.h, 0, GL_RGBA,
                 texelType, frameBuffer.data());

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, size.w, size.h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)size.w, 0.f, (float)size.h, -1.f, 1.f);

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)size.h, 0.f);
        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)size.w, (float)size.h, 0.f);
        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)size.w, 0.f, 0.f);
    }
    glEnd();
}

/* Put the window in the center of user's monitor*/
void MainWindow::PutInCenter()
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
