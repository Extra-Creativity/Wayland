#include <iostream>
#include "Renderer.h"
#include "SceneManager.h"

using namespace Wayland;

int main(int argc, char** argv) {
     
    string sceneSrc =
        // R"(C:\Users\60995\Desktop\Grad_Design\pbrt-v3-scenes\barcelona-pavilion\pavilion-day.pbrt)";
        R"(C:\Users\60995\Desktop\Grad_Design\Wayland\scene\cornell-box\cbox.pbrt)";

     try
     {
         Wayland::Renderer app({ 1920, 1080 }, sceneSrc);

         //app.scene.printScene();

         //app.Run();
     }
     catch (const std::exception &e)
     {
         SPDLOG_ERROR("{}", e.what());
         return 1;
     }
    return 0;
}