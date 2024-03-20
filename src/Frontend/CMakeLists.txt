find_package(OpenGL REQUIRED)

set(TARGET_NAME wayland)

set(
  SOURCE
    Wayland.cpp
    MainWindow.cpp
    Renderer.cpp
    SceneManager.cpp
    OptixManager.cpp
    Camera.cpp
    Shape.cpp
    PrintUtils.cpp
)
set(
  HEADERS
    MainWindow.h
    Renderer.h
    SceneManager.h
    OptixManager.h
    Camera.h
    Shape.h
    PrintUtils.h
    PBRTv3_SceneCommon.h
)
  
add_executable(${TARGET_NAME} ${SOURCE} ${HEADERS})

target_compile_definitions(
    ${TARGET_NAME} 
    PRIVATE NEED_AUTO_PROGRAM_CONFIG
    PRIVATE GLM_DIR=R\"\(${PROJECT_SOURCE_DIR}/build/build/_deps/glm-src\)\"
    PRIVATE OPTIX_DIR=R\"\($ENV{OptiX_ROOT}/include\)\"
    PRIVATE UTILS_INC_PATH=R\"\(${PROJECT_SOURCE_DIR}/src/Backend/Optix/Utils\)\"
)

target_include_directories(
    ${TARGET_NAME} 
    PRIVATE ${PROJECT_SOURCE_DIR}/src/Backend/OptiX 
    PRIVATE ${PROJECT_SOURCE_DIR}/src/Backend/OptiX/Utils
)

target_link_libraries(${TARGET_NAME} PRIVATE 
    OptiX-Core
    minipbrt
    glfw 
    spdlog
    glm
    ${OPENGL_gl_LIBRARY}
)