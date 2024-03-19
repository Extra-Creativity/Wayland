find_package(OpenGL REQUIRED)

set(TARGET_NAME wayland)

set(
  SOURCE
    Wayland.cpp
    MainWindow.cpp
    Renderer.cpp
    SceneManager.cpp
    Camera.cpp
    Shape.cpp
    PrintUtils.cpp
)
set(
  HEADERS
    MainWindow.h
    Renderer.h
    SceneManager.h
    Camera.h
    Shape.h
    PrintUtils.h
    PBRTv3_SceneCommon.h
)
  
add_executable(${TARGET_NAME} ${SOURCE} ${HEADERS})
target_compile_definitions(${TARGET_NAME} PRIVATE NEED_AUTO_PROGRAM_CONFIG)

target_include_directories(
    ${TARGET_NAME} 
    PRIVATE ${PROJECT_SOURCE_DIR}/src/Backend/Optix 
    PRIVATE ${PROJECT_SOURCE_DIR}/src/Backend/Optix/Utils
)

target_link_libraries(${TARGET_NAME} PRIVATE 
    Optix-Core
    minipbrt
    glfw 
    spdlog
    glm
    ${OPENGL_gl_LIBRARY}
)