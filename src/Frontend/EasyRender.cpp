#include <string>
#include <iostream>

#include "Core/Renderer.h"
#include "Core/SceneManager.h"

using namespace EasyRender;

std::string sceneSrc =
    R"(..\..\..\scene\cornell-box\cbox.pbrt)";
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
     //R"(..\..\..\scene\kitchen\kitchen.pbrt)";


int main(int argc, char **argv)
{
    try
    {
        RendererSetting mySet(argc, argv);
        mySet.SetScenePath(sceneSrc);
        mySet.SetProgram(ProgramType::BDPT);
        mySet.SetResolution(1000, 1000);
        mySet.SetOutputPath(R"(C:\Users\60995\Desktop\Grad_Design\EasyRender-Results\result\)");

        EasyRender::Renderer app(mySet);
        app.Run();
    }
    catch (const std::exception &e)
    {
        SPDLOG_ERROR("{}", e.what());
        return 1;
    }
    return 0;
}
