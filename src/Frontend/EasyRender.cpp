#include <iostream>

#include "Core/Renderer.h"
#include "Core/SceneManager.h"

using namespace EasyRender;

const std::string sceneSrc =
    // R"(C:\Users\60995\Desktop\Grad_Design\pbrt-v3-scenes\barcelona-pavilion\pavilion-day.pbrt)";
    R"(..\..\..\scene\cornell-box\cbox.pbrt)";

int main(int argc, char **argv)
{
    ProgramType program = ProgramType::Normal;
    try
    {
        EasyRender::Renderer app({ 1000, 1000 }, program, sceneSrc);
        // app.scene.PrintScene();
        app.Run();
    }
    catch (const std::exception &e)
    {
        SPDLOG_ERROR("{}", e.what());
        return 1;
    }
    return 0;
}