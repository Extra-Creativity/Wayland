#include <iostream>

#include "Core/Renderer.h"
#include "Core/SceneManager.h"

using namespace EasyRender;

std::string sceneSrc =
    //R"(C:\Users\60995\Desktop\Grad_Design\pbrt-v3-scenes\barcelona-pavilion\pavilion-day.pbrt)";
    //R"(..\..\..\scene\cornell-box\cbox.pbrt)";
    //R"(..\..\..\scene\cornell-box\cbox-texture.pbrt)";
    //R"(..\..\..\scene\dragon\dragon.pbrt)";
    //R"(..\..\..\scene\dragon\dragon-color.pbrt)";
    //R"(..\..\..\scene\dragon\dragon-glass.pbrt)";
    //R"(..\..\..\scene\dragon\dragon-metal-silver.pbrt)";
    //R"(..\..\..\scene\dragon\dragon-metal-black.pbrt)";
    //R"(..\..\..\scene\veach-mis\mis-lambert.pbrt)";
    //R"(..\..\..\scene\veach-mis\mis.pbrt)";
    //R"(..\..\..\scene\veach-bidir\bidir.pbrt)";
    //R"(..\..\..\scene\water-caustic\water.pbrt)";
    //R"(..\..\..\scene\veach-ajar\ajar.pbrt)";
     //R"(..\..\..\scene\staircase\staircase.pbrt)";
     //R"(..\..\..\scene\glass-of-water\glass-of-water.pbrt)";
     R"(..\..\..\scene\kitchen\kitchen.pbrt)";


int main(int argc, char **argv)
{
    ProgramType program = ProgramType::PathTracing;
    try
    {
        EasyRender::Renderer app({ 1280, 720 }, program, sceneSrc);
        app.Run();
    }
    catch (const std::exception &e)
    {
        SPDLOG_ERROR("{}", e.what());
        return 1;
    }
    return 0;
}