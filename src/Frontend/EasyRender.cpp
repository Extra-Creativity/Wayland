#include <iostream>

#include "Core/Renderer.h"
#include "Core/SceneManager.h"

using namespace EasyRender;

const std::string sceneSrc =
    //R"(C:\Users\60995\Desktop\Grad_Design\pbrt-v3-scenes\barcelona-pavilion\pavilion-day.pbrt)";
    //R"(..\..\..\scene\cornell-box\cbox.pbrt)";
    //R"(..\..\..\scene\cornell-box\cbox-texture.pbrt)";
    //R"(..\..\..\scene\dragon\dragon.pbrt)";
    //R"(..\..\..\scene\veach-mis\mis-lambert.pbrt)";
    //R"(..\..\..\scene\veach-mis\mis.pbrt)";
     R"(..\..\..\scene\veach-bidir\bidir.pbrt)";
    //R"(..\..\..\scene\dragon\dragon.pbrt)";

int main(int argc, char **argv)
{
    ProgramType program = ProgramType::PathTracing;
    try
    {
        EasyRender::Renderer app({ 1152, 864 }, program, sceneSrc);
        //EasyRender::Renderer app({ 1200, 1000 }, program, sceneSrc);
        app.Run();
    }
    catch (const std::exception &e)
    {
        SPDLOG_ERROR("{}", e.what());
        return 1;
    }
    return 0;
}