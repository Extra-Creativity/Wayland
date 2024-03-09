FILE(GLOB PhongSrc *.cpp)
FILE(GLOB PhongHeaders *.h)

add_executable(Example_Phong ${PhongSrc} ${PhongHeaders})
target_link_libraries(Example_Phong PRIVATE Example_Common Example_Common_Base)
target_include_directories(Example_Phong PRIVATE ../Common)
target_compile_definitions(Example_Phong 
    PRIVATE SHADER_PATH=R\"\(${CMAKE_CURRENT_SOURCE_DIR}/Phong.cu\)\"
    PRIVATE MODEL_PATH=R\"\(${CMAKE_CURRENT_SOURCE_DIR}/data/Sucrose.pmx\)\"
    PRIVATE IMAGE_PATH=R\"\(${CMAKE_CURRENT_SOURCE_DIR}/image.jpg\)\"
    PRIVATE NEED_AUTO_PROGRAM_CONFIG)

# add_library(PhongShader Phong.cu)