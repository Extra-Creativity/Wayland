set_xmakever("2.8.2")
set_project("Wayland")

add_rules("mode.debug", "mode.release", "mode.releasedbg")
add_rules("mode.debug-dev", "mode.releasedbg-dev", "mode.release-dev")
if is_mode("release") then 
    set_policy("build.optimization.lto", true)
end

add_rules("plugin.vsxmake.autoupdate")
add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})

set_languages("cxxlatest")
add_cuflags("--std c++20")
set_policy("build.warning", true)
set_warnings("all")

set_encodings("utf-8")
set_exceptions("cxx")

-- cudadevrt -> to use thrust
-- cuda -> to use driver lib
add_requires("cuda", {configs={utils={"cuda", "cudadevrt"}}})
add_requires("spdlog", {configs={std_format=true}})
add_requires("optix", "glm")
add_packages("cuda", "optix", "spdlog", "glm")

includes("src")