#include <iostream>
#include "Renderer.h"

using namespace Wayland;

int main(int argc, char** argv) {
     
     try
     {
         Wayland::Renderer app({ 1920, 1080 });
         app.Run();
     }
     catch (const std::exception& e)
     {
         std::cerr << "Caught Exception: " <<  e.what() << std::endl;
         return 1;
	 }
    return 0;
}