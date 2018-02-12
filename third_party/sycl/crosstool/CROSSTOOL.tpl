major_version: "local"
minor_version: ""
default_target_cpu: "same_as_host"

default_toolchain {
  cpu: "k8"
  toolchain_identifier: "local_linux"
}

default_toolchain {
  cpu: "arm"
  toolchain_identifier: "local_linux"
}

default_toolchain {
  cpu: "armeabi"
  toolchain_identifier: "%{ARM_TARGET}%"
}

toolchain {
  abi_version: "local"
  abi_libc_version: "local"
  builtin_sysroot: ""
  compiler: "compiler"
  host_system_name: "local"
  needsPic: true
  supports_gold_linker: false
  supports_incremental_linker: false
  supports_fission: false
  supports_interface_shared_objects: false
  supports_normalizing_ar: false
  supports_start_end_lib: false
  supports_thin_archives: false
  target_libc: "local"
  target_cpu: "local"
  target_system_name: "local"
  toolchain_identifier: "local_linux"

  tool_path { name: "ar" path: "/usr/bin/ar" }
  tool_path { name: "compat-ld" path: "/usr/bin/ld" }
  tool_path { name: "cpp" path: "/usr/bin/cpp" }
  tool_path { name: "dwp" path: "/usr/bin/dwp" }
  tool_path { name: "gcc" path: "%{COMPUTECPP_ROOT_DIR}%/bin/compute" }
  tool_path { name: "g++" path: "%{COMPUTECPP_ROOT_DIR}%/bin/compute++" }
  tool_path { name: "gcov" path: "/usr/bin/gcov" }
  tool_path { name: "ld" path: "/usr/bin/ld" }
  tool_path { name: "nm" path: "/usr/bin/nm" }
  tool_path { name: "objcopy" path: "/usr/bin/objcopy" }
  tool_path { name: "objdump" path: "/usr/bin/objdump" }
  tool_path { name: "strip" path: "/usr/bin/strip" }

  cxx_builtin_include_directory: "/usr/lib/gcc/"
  cxx_builtin_include_directory: "/usr/lib"
  cxx_builtin_include_directory: "/usr/lib64"
  cxx_builtin_include_directory: "/usr/local/include"
  cxx_builtin_include_directory: "/usr/include"
  cxx_builtin_include_directory: "%{COMPUTECPP_ROOT_DIR}%"

  cxx_flag: "-std=c++11"
  cxx_flag: "-isystem"
  cxx_flag: "%{PYTHON_INCLUDE_PATH}%"
  cxx_flag: "-isystem"
  cxx_flag: "%{COMPUTECPP_ROOT_DIR}%/include"
  cxx_flag: "-fsycl-ih-last"
  cxx_flag: "-sycl-driver"
  # Serialisation of memory operations is required on some platforms
  #cxx_flag: "-no-serial-memop"
  cxx_flag: "-Xclang"
  cxx_flag: "-cl-denorms-are-zero"
  cxx_flag: "-Xclang"
  cxx_flag: "-cl-fp32-correctly-rounded-divide-sqrt"
  cxx_flag: "-Xclang"
  cxx_flag: "-cl-mad-enable"
  cxx_flag: "-sycl-target"
  cxx_flag: "%{BITCODE_FORMAT}%"
  cxx_flag: "-DTENSORFLOW_USE_SYCL=1"
  cxx_flag: "-DEIGEN_USE_SYCL=1"
  cxx_flag: "-DEIGEN_HAS_C99_MATH=1"
  cxx_flag: "-DEIGEN_HAS_CXX11_MATH=1"

  unfiltered_cxx_flag: "-Wno-builtin-macro-redefined"
  unfiltered_cxx_flag: "-D__DATE__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIMESTAMP__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIME__=\"redacted\""
  unfiltered_cxx_flag: "-no-canonical-prefixes"

  compiler_flag: "-U_FORTIFY_SOURCE"
  compiler_flag: "-D_FORTIFY_SOURCE=1"
  compiler_flag: "-fstack-protector"
  compiler_flag: "-fPIE"
  compiler_flag: "-fno-omit-frame-pointer"
  compiler_flag: "-Wall"

  linker_flag: "-lstdc++"
  linker_flag: "-no-canonical-prefixes"
  linker_flag: "-Wl,-no-as-needed"
  linker_flag: "-Wl,-z,relro,-z,now"
  linker_flag: "-Wl,--build-id=md5"
  linker_flag: "-Wl,--hash-style=gnu"

  objcopy_embed_flag: "-I"
  objcopy_embed_flag: "binary"

  compilation_mode_flags {
    mode: FASTBUILD
    compiler_flag: "-O0"
  }

  compilation_mode_flags {
    mode: DBG
    compiler_flag: "-g"
  }

  compilation_mode_flags {
    mode: OPT
    compiler_flag: "-g0"
    compiler_flag: "-O2"
    compiler_flag: "-DNDEBUG"
    compiler_flag: "-ffunction-sections"
    compiler_flag: "-fdata-sections"
    linker_flag: "-Wl,--gc-sections"
  }

  linking_mode_flags { mode: DYNAMIC }
}

toolchain {
  abi_version: "armeabi"
  abi_libc_version: "armeabi"
  builtin_sysroot: ""
  compiler: "compiler"
  host_system_name: "armeabi"
  needsPic: true
  supports_gold_linker: false
  supports_incremental_linker: false
  supports_fission: false
  supports_interface_shared_objects: false
  supports_normalizing_ar: false
  supports_start_end_lib: false
  supports_thin_archives: false
  target_libc: "armeabi"
  target_cpu: "armeabi"
  target_system_name: "armeabi"
  toolchain_identifier: "%{ARM_TARGET}%"

  tool_path { name: "ar" path: "%{ARM_COMPILER_PATH}%/bin/%{ARM_TARGET}%-ar" }
  tool_path { name: "compat-ld" path: "/bin/false" }
  tool_path { name: "cpp" path: "%{ARM_COMPILER_PATH}%/bin/%{ARM_TARGET}%-cpp" }
  tool_path { name: "dwp" path: "%{ARM_COMPILER_PATH}%/bin/%{ARM_TARGET}%-dwp" }
  tool_path { name: "gcc" path: "%{COMPUTECPP_ROOT_DIR}%/bin/compute" }
  tool_path { name: "g++" path: "%{COMPUTECPP_ROOT_DIR}%/bin/compute++" }
  tool_path { name: "gcov" path: "%{ARM_COMPILER_PATH}%/bin/%{ARM_TARGET}%-gcov" }
  tool_path { name: "ld" path: "%{ARM_COMPILER_PATH}%/bin/%{ARM_TARGET}%-ld" }
  tool_path { name: "nm" path: "%{ARM_COMPILER_PATH}%/bin/%{ARM_TARGET}%-nm" }
  tool_path { name: "objcopy" path: "%{ARM_COMPILER_PATH}%/bin/%{ARM_TARGET}%-objcopy" }
  tool_path { name: "objdump" path: "%{ARM_COMPILER_PATH}%/bin/%{ARM_TARGET}%-objdump" }
  tool_path { name: "strip" path: "%{ARM_COMPILER_PATH}%/bin/%{ARM_TARGET}%-strip" }

  cxx_builtin_include_directory: "%{ARM_COMPILER_PATH}%"
  cxx_builtin_include_directory: "%{COMPUTECPP_ROOT_DIR}%"
  cxx_builtin_include_directory: "%{OPENCL_INCLUDE_DIR}%"
  cxx_builtin_include_directory: "%{PYTHON_INCLUDE_PATH}%"

  compiler_flag: "-target"
  compiler_flag: "%{ARM_TARGET}%"
  compiler_flag: "--gcc-toolchain=%{ARM_COMPILER_PATH}%"
  compiler_flag: "--sysroot=%{ARM_COMPILER_PATH}%/%{ARM_TARGET}%/libc"

  cxx_flag: "-std=c++11"
  cxx_flag: "-isystem"
  cxx_flag: "%{PYTHON_INCLUDE_PATH}%"
  cxx_flag: "-isystem"
  cxx_flag: "%{COMPUTECPP_ROOT_DIR}%/include"
  cxx_flag: "-isystem"
  cxx_flag: "%{OPENCL_INCLUDE_DIR}%"
  cxx_flag: "-fsycl-ih-last"
  cxx_flag: "-sycl-driver"
  # Serialisation of memory operations is required on some platforms
  #cxx_flag: "-no-serial-memop"
  cxx_flag: "-Xclang"
  cxx_flag: "-cl-denorms-are-zero"
  cxx_flag: "-Xclang"
  cxx_flag: "-cl-fp32-correctly-rounded-divide-sqrt"
  cxx_flag: "-Xclang"
  cxx_flag: "-cl-mad-enable"
  cxx_flag: "-sycl-target"
  cxx_flag: "%{BITCODE_FORMAT}%"
  cxx_flag: "-DTENSORFLOW_USE_SYCL=1"
  cxx_flag: "-DEIGEN_USE_SYCL=1"
  cxx_flag: "-DEIGEN_HAS_C99_MATH=1"
  cxx_flag: "-DEIGEN_HAS_CXX11_MATH=1"

  unfiltered_cxx_flag: "-Wno-builtin-macro-redefined"
  unfiltered_cxx_flag: "-D__DATE__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIMESTAMP__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIME__=\"redacted\""
  unfiltered_cxx_flag: "-no-canonical-prefixes"

  compiler_flag: "-U_FORTIFY_SOURCE"
  compiler_flag: "-D_FORTIFY_SOURCE=1"
  compiler_flag: "-fstack-protector"
  compiler_flag: "-fPIE"
  compiler_flag: "-fno-omit-frame-pointer"
  compiler_flag: "-Wall"

  linker_flag: "-target"
  linker_flag: "%{ARM_TARGET}%"
  linker_flag: "--gcc-toolchain=%{ARM_COMPILER_PATH}%"
  linker_flag: "--sysroot=%{ARM_COMPILER_PATH}%/%{ARM_TARGET}%/libc"
  linker_flag: "-lstdc++"
  linker_flag: "-no-canonical-prefixes"
  linker_flag: "-Wl,-z,relro,-z,now"
  linker_flag: "-Wl,--build-id=md5"
  linker_flag: "-Wl,--hash-style=gnu"

  objcopy_embed_flag: "-I"
  objcopy_embed_flag: "binary"

  compilation_mode_flags {
    mode: FASTBUILD
    compiler_flag: "-O0"
  }

  compilation_mode_flags {
    mode: DBG
    compiler_flag: "-g"
  }

  compilation_mode_flags {
    mode: OPT
    compiler_flag: "-g0"
    compiler_flag: "-O2"
    compiler_flag: "-DNDEBUG"
    compiler_flag: "-ffunction-sections"
    compiler_flag: "-fdata-sections"
    linker_flag: "-Wl,--gc-sections"
  }

  linking_mode_flags { mode: DYNAMIC }
}
