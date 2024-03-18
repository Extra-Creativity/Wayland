#include <iostream>
#include "Renderer.h"
#include "SceneManager.h"

using namespace Wayland;

int main(int argc, char** argv) {
     
    string sceneSrc =
        R"(C:\Users\60995\Desktop\Grad_Design\pbrt-v3-scenes\barcelona-pavilion\pavilion-day.pbrt)";

     try
     {
         Wayland::Renderer app({ 1920, 1080 });

         SceneManager scene(sceneSrc);
         
         scene.printScene();

         //app.Run();
     }
     catch (const std::exception& e)
     {
         std::cerr << "Caught Exception: " <<  e.what() << std::endl;
         return 1;
	 }
    return 0;
}