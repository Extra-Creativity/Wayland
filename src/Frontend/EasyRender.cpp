#include <iostream>

#include "Core/Renderer.h"
#include "Core/SceneManager.h"

using namespace EasyRender;

int main(int argc, char** argv) {
     
    string sceneSrc =
        // R"(C:\Users\60995\Desktop\Grad_Design\pbrt-v3-scenes\barcelona-pavilion\pavilion-day.pbrt)";
        R"(..\..\..\scene\cornell-box\cbox.pbrt)";

    string programSrc =
        //R"(..\..\..\src\Frontend\Programs\TestExamples\Simple\Simple.cu)";
        //R"(..\..\..\src\Frontend\Programs\TestExamples\Depth\Depth.cu)";
        //R"(..\..\..\src\Frontend\Programs\TestExamples\Mesh\Mesh.cu)";
        R"(..\..\..\src\Frontend\Programs\TestExamples\Color\Color.cu)";

     try
     {
         EasyRender::Renderer app({ 1000, 1000 }, sceneSrc, programSrc);
         //app.scene.printScene();
         app.Run();
     }
     catch (const std::exception &e)
     {
         SPDLOG_ERROR("{}", e.what());
         return 1;
     }
    return 0;
}