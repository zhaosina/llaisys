add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

option("cpu-openmp")
    set_default(true)
    set_showmenu(true)
    set_description("Enable OpenMP acceleration for CPU kernels")
option_end()

option("cpu-avx2")
    if is_plat("linux") and is_arch("x64", "x86_64") then
        set_default(true)
    else
        set_default(false)
    end
    set_showmenu(true)
    set_description("Enable AVX2/FMA acceleration for CPU kernels")
option_end()

option("cpu-openblas")
    set_default(false)
    set_showmenu(true)
    set_description("Enable OpenBLAS acceleration for CPU linear kernels")
option_end()

if has_config("cpu-openmp") and is_plat("linux") then
    add_cxflags("-fopenmp")
    add_ldflags("-fopenmp")
end

if has_config("cpu-avx2") and is_plat("linux") and is_arch("x64", "x86_64") then
    add_defines("LLAISYS_CPU_AVX2=1")
    add_cxflags("-mavx2", "-mfma")
end

if has_config("cpu-openblas") then
    add_defines("LLAISYS_USE_OPENBLAS=1")
end

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    if has_config("nv-gpu") then
        set_languages("cxx17", "cuda")
    else
        set_languages("cxx17")
    end
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    if has_config("nv-gpu") then
        add_files("src/llaisys/cuda_link.cu")
    end
    set_installdir(".")

    if has_config("cpu-openmp") and is_plat("linux") then
        add_shflags("-Wl,--push-state,--no-as-needed", "-lgomp", "-Wl,--pop-state")
    end

    if has_config("cpu-openblas") then
        add_links("openblas")
    end

    if has_config("nv-gpu") then
        add_links("cudart")
        add_links("cudadevrt")
        add_links("cublas")
        add_shflags("-Wl,--push-state,--no-as-needed", "-lcudart", "-lcudadevrt", "-lcublas", "-Wl,--pop-state")
    end

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()
