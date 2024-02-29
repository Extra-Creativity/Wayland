cmake_minimum_required(VERSION 3.17)
project(Wayland)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit)
link_libraries(CUDA::cuda_driver)
link_libraries(CUDA::cudart)

file(GLOB LibPaths build/vcpkg/pkgs/*/share/*)

set(CMAKE_PREFIX_PATH "${LibPaths}")

include_directories(SYSTEM
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include"
    "D:/Work/CS/Graphics/Optix/include")

add_definitions(-DSPDLOG_USE_STD_FORMAT)
find_package(spdlog REQUIRED)
find_package(glm REQUIRED)

# Enable UTF-8 and warnings.
if(MSVC)
    add_compile_options(/utf-8 /W3 /wd4996) # 4996 is warning from spdlog.
else()
    add_compile_options(-fexec-charset=UTF-8 -finput-charset=UTF-8 -Wall)
endif()

option(DEBUG OFF)
if(DEBUG)
    add_definitions(-DERROR_DEBUG -DNEED_VALIDATION_MODE -DNEED_IN_RANGE_CHECK 
                  -DNEED_VALID_DEVICE_POINTER_CHECK -DNEED_SAFE_INT_CHECK)
    message("Enabling debug mode...")
endif()

option(RELEASE OFF)
if(RELEASE)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # enable LTO
endif()

add_subdirectory(src)
add_subdirectory(private)