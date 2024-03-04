FILE(GLOB HostUtilsSrc Utils/HostUtils/*.cpp)
FILE(GLOB HostUtilsHeaders Utils/HostUtils/*.h)
FILE(GLOB UniUtilsHeaders Utils/UniUtils/*.h)
add_library(Optix-HostUtils STATIC ${HostUtilsSrc} ${HostUtilsHeaders} ${UniUtilsHeaders})
target_link_libraries(Optix-HostUtils PRIVATE spdlog::spdlog)

FILE(GLOB CoreSrc Core/*.cpp)
FILE(GLOB CoreHeaders Core/*.h)
add_library(Optix-Core STATIC ${CoreSrc} ${CoreHeaders})
target_include_directories(Optix-Core PUBLIC Core Utils)
target_link_libraries(Optix-Core PUBLIC Optix-HostUtils spdlog::spdlog glm::glm)

option(OPTIX_DEV OFF)
option(NEED_RE2 "Re2 is a fast regex library; otherwise the standard library will be used, which is much slower." OFF)
if(OPTIX_DEV)
    if(NEED_RE2)
        set(ABSL_ENABLE_INSTALL ON)
        set(ABSL_PROPAGATE_CXX_STD ON)
        FetchContent_Declare(
            absl
            GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
            GIT_TAG 20240116.1
            OVERRIDE_FIND_PACKAGE
        )
        FetchContent_MakeAvailable(absl)

        FetchContent_Declare(
            re2
            GIT_REPOSITORY    https://github.com/google/re2.git
            GIT_TAG           2024-02-01
        )
        FetchContent_MakeAvailable(re2)
        target_compile_definitions(Optix-Core PRIVATE NEED_RE2)
        target_link_libraries(Optix-Core PRIVATE re2::re2)
    endif()
    target_compile_definitions(Optix-Core PRIVATE NEED_AUTO_PROGRAM_CONFIG)
endif()