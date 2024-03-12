FILE(GLOB PTSrc *.cpp)
FILE(GLOB PTHeaders *.h)

add_executable(Example_PT ${PTSrc} ${PTHeaders})
target_link_libraries(Example_PT PRIVATE Example_Common Example_Common_Base)
target_include_directories(Example_PT PRIVATE ../Common)
target_compile_definitions(Example_PT 
    PRIVATE SHADER_PATH=R\"\(${CMAKE_CURRENT_SOURCE_DIR}/PathTracer.cu\)\"
    PRIVATE MODEL_PATH=R\"\(${CMAKE_CURRENT_SOURCE_DIR}/data/cornell.obj\)\"
    PRIVATE IMAGE_PATH=R\"\(${CMAKE_CURRENT_SOURCE_DIR}/image.jpg\)\"
    PRIVATE NEED_AUTO_PROGRAM_CONFIG)
