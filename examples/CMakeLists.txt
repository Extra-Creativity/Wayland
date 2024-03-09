FetchContent_Declare(
  assimp
  GIT_REPOSITORY    https://github.com/assimp/assimp.git
  GIT_TAG           v5.3.1
)

FetchContent_MakeAvailable(assimp)

FILE(GLOB CommonBasicSrc Common/*.cpp)
FILE(GLOB CommonBasicHeaders Common/*.h)
add_library(Example_Common_Base STATIC ${CommonBasicSrc} ${CommonBasicHeaders})
target_compile_definitions(Example_Common_Base 
  PRIVATE GLM_DIR=R\"\(${CMAKE_SOURCE_DIR}/build/_deps/glm-src\)\"
  PRIVATE OPTIX_DIR=R\"\($ENV{OptiX_ROOT}/include\)\"
  PRIVATE DEVICE_UTILS_LIB_PATH=R\"\($<TARGET_FILE:Optix-DeviceUtils>\)\"
  PRIVATE UTILS_INC_PATH=R\"\(${CMAKE_SOURCE_DIR}/src/Backend/Optix/Utils\)\"
  )
target_link_libraries(Example_Common_Base PRIVATE Optix-Core)
target_include_directories(Example_Common_Base PRIVATE ../src/Backend)

FILE(GLOB_RECURSE CommonSrc Common/*/*.cpp)
FILE(GLOB_RECURSE CommonHeaders Common/*/*.h)
list(REMOVE_ITEM CommonHeaders Common/DeviceUtils/*.h)

add_library(Example_Common STATIC ${CommonSrc} ${CommonHeaders})
target_include_directories(Example_Common PUBLIC ../src/Backend PRIVATE Common)
target_link_libraries(Example_Common PUBLIC Optix-Core assimp::assimp Example_Common_Base)

add_subdirectory(Phong)