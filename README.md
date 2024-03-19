# Wayland
A renderer in development.

## Dependencies
+ [spdlog](https://github.com/gabime/spdlog), please specify `-DSPDLOG_USE_STD_FORMAT` when built with CMake. 
+ [re2](https://github.com/google/re2)
+ [glm](https://github.com/g-truc/glm)
+ OptiX, CUDA for OptiX backend.

We also provide `vcpkg.json` for those who use vcpkg as package manager, but you may specify cmake options yourself when you install packages.
