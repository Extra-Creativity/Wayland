find_package(OpenGL REQUIRED)

# https://cloud.tencent.com/developer/article/2300671
macro(source_group_by_dir source_files)
  if(MSVC)
      set(sgbd_cur_dir ${CMAKE_CURRENT_SOURCE_DIR})
      foreach(sgbd_file ${${source_files}})
          string(REGEX REPLACE ${sgbd_cur_dir}/\(.*\) \\1 sgbd_fpath ${sgbd_file})
          string(REGEX REPLACE "\(.*\)/.*" \\1 sgbd_group_name ${sgbd_fpath})
          string(COMPARE EQUAL ${sgbd_fpath} ${sgbd_group_name} sgbd_nogroup)
          string(REPLACE "/" "\\" sgbd_group_name ${sgbd_group_name})
          if(sgbd_nogroup)
              set(sgbd_group_name "\\")
          endif(sgbd_nogroup)
          source_group(${sgbd_group_name} FILES ${sgbd_file})
      endforeach(sgbd_file)
  endif(MSVC)
endmacro(source_group_by_dir)

set(TARGET_NAME wayland)

file(GLOB_RECURSE HEADERS *.h)
file(GLOB_RECURSE SOURCE *.cpp *.cu)
set(all_files ${HEADERS} ${SOURCE})
source_group_by_dir(all_files)

add_executable(${TARGET_NAME} ${all_files})

target_compile_definitions(
    ${TARGET_NAME} 
    PRIVATE NEED_AUTO_PROGRAM_CONFIG
    PRIVATE GLM_DIR=R\"\(${PROJECT_SOURCE_DIR}/build/build/_deps/glm-src\)\"
    PRIVATE OPTIX_DIR=R\"\($ENV{OptiX_ROOT}/include\)\"
    PRIVATE UTILS_INC_PATH=R\"\(${PROJECT_SOURCE_DIR}/src/Backend/Optix/Utils\)\"
    PRIVATE FRONT_PATH=R\"\(${PROJECT_SOURCE_DIR}/src/Frontend\)\"
)

target_include_directories(
    ${TARGET_NAME} 
    PRIVATE ${PROJECT_SOURCE_DIR}/src/Backend/OptiX 
    PRIVATE ${PROJECT_SOURCE_DIR}/src/Backend/OptiX/Utils
    PRIVATE ${PROJECT_SOURCE_DIR}/src/Frontend   
)

target_link_libraries(${TARGET_NAME} PRIVATE 
    OptiX-Core
    minipbrt
    glfw 
    spdlog
    glm
    ${OPENGL_gl_LIBRARY}
)