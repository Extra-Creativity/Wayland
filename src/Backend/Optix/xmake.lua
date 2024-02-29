if is_mode("debug") or is_mode("debug-dev") then
    add_defines("ERROR_DEBUG", "NEED_VALIDATION_MODE", "NEED_IN_RANGE_CHECK", 
                "NEED_VALID_DEVICE_POINTER_CHECK", "NEED_SAFE_INT_CHECK")
end

add_requires("re2", {optional=true})
option("need-re2")
    set_values(true, false)
    set_default(false)
    set_showmenu(true)

rule("need-host-utils")
    on_load(function (target) 
        target:add("deps", "HostUtils")
        target:add("includedirs", path.join(os.scriptdir(), "Utils"))
    end)

rule("need-optix-core")
    add_deps("need-host-utils")
    on_load(function (target) 
        target:add("deps", "Optix-Core")
        target:add("includedirs", path.join(os.scriptdir()))
    end)

rule("mode.debug-dev")
    on_load(function (target) 
        if is_mode("debug-dev") then
            target:set("symbols", "debug")
            target:set("optimize", "none")
            target:add("defines", "NEED_AUTO_PROGRAM_CONFIG")
            if get_config("need-re2") then
                target:add("packages", "re2")
                target:add("defines", "NEED_RE2")
            end
        end
    end)
rule_end()

rule("mode.releasedbg-dev")
    on_load(function (target) 
        if is_mode("releasedbg-dev") then
            target:set("symbols", "debug")
            target:set("optimize", "fastest")
            target:set("strip", "all")
            target:add("defines", "NEED_AUTO_PROGRAM_CONFIG")
            if get_config("need-re2") then
                target:add("packages", "re2")
                target:add("defines", "NEED_RE2")
            end
        end
    end)
rule_end()

rule("mode.release-dev")
    on_load(function (target) 
        if is_mode("release-dev") then
            target:set("symbols", "hidden")
            target:set("optimize", "fastest")
            target:set("strip", "all")
            target:add("defines", "NEED_AUTO_PROGRAM_CONFIG")
            if get_config("need-re2") then
                target:add("packages", "re2")
                target:add("defines", "NEED_RE2")
            end
            target:set("policy", "build.optimization.lto", true)
        end
    end)
rule_end()

target("HostUtils")
    set_kind("static")
    add_headerfiles("Utils/HostUtils/*.h")
    add_files("Utils/HostUtils/*.cpp")

target("Optix-Core")
    set_kind("static")
    add_rules("need-host-utils")
    add_headerfiles("Core/*.h")
    add_files("Core/*.cpp")