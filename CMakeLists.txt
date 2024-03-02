cmake_minimum_required(VERSION 3.17)
project(Wayland)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BINARY_DIR build)

find_package(CUDAToolkit)
link_libraries(CUDA::cuda_driver)
link_libraries(CUDA::cudart)

set(CMAKE_PREFIX_PATH "${LibPaths}")

include(FetchContent)
include(ExternalProject)

set(SPDLOG_USE_STD_FORMAT ON)
if(MSVC)
    add_compile_options(/wd4996)
endif()

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY    https://github.com/gabime/spdlog.git
  GIT_TAG           v1.13.0
)

FetchContent_Declare(
  glm
  GIT_REPOSITORY    https://github.com/g-truc/glm.git
  GIT_TAG           1.0.1
)

FetchContent_MakeAvailable(spdlog glm)
include_directories(SYSTEM
    "${CUDAToolkit_INCLUDE_DIRS}"
    "$ENV{OptiX_ROOT}/include")

add_definitions(-DSPDLOG_USE_STD_FORMAT)

# Enable UTF-8 and warnings.
if(MSVC)
    add_compile_options(/utf-8 /W3) # 4996 is warning from spdlog.
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