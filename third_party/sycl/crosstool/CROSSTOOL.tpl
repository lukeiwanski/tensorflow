major_version: "local"
minor_version: ""
default_target_cpu: "same_as_host"

default_toolchain {
  cpu: "k8"
  toolchain_identifier: "local_linux"
}
default_toolchain {
  cpu: "armeabi"
  toolchain_identifier: "arm-linux-gnueabihf"
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
  tool_path { name: "gcc" path: "/usr/bin/gcc" }
  tool_path { name: "g++" path: "%{COMPUTECPP_ROOT_DIR}%/bin/compute++" }
  cxx_flag: "-std=c++11"
  linker_flag: "-Wl,-no-as-needed"
  linker_flag: "-lstdc++"
  linker_flag: "-B/usr/bin/"

  # TODO(bazel-team): In theory, the path here ought to exactly match the path
  # used by gcc. That works because bazel currently doesn't track files at
  # absolute locations and has no remote execution, yet. However, this will need
  # to be fixed, maybe with auto-detection?
  cxx_builtin_include_directory: "/usr/lib/gcc/"
  cxx_builtin_include_directory: "/usr/lib"
  cxx_builtin_include_directory: "/usr/lib64"
  cxx_builtin_include_directory: "/usr/local/include"
  cxx_builtin_include_directory: "/usr/include"

  cxx_builtin_include_directory: "%{COMPUTECPP_ROOT_DIR}%/include"
  cxx_builtin_include_directory: "%{python_lib_path}"

  tool_path { name: "gcov" path: "/usr/bin/gcov" }

  # C(++) compiles invoke the compiler (as that is the one knowing where
  # to find libraries), but we provide LD so other rules can invoke the linker.
  tool_path { name: "ld" path: "/usr/bin/ld" }

  tool_path { name: "nm" path: "/usr/bin/nm" }
  tool_path { name: "objcopy" path: "/usr/bin/objcopy" }
  objcopy_embed_flag: "-I"
  objcopy_embed_flag: "binary"
  tool_path { name: "objdump" path: "/usr/bin/objdump" }
  tool_path { name: "strip" path: "/usr/bin/strip" }
  cxx_flag: "-std=c++11"
  cxx_flag: "-fsycl-ih-last"
  cxx_flag: "-sycl-driver"
  cxx_flag: "-no-serial-memop"
  cxx_flag: "-Xclang"
  cxx_flag: "-cl-denorms-are-zero"
  cxx_flag: "-Xclang"
  cxx_flag: "-cl-fp32-correctly-rounded-divide-sqrt"
  cxx_flag: "-Xclang"
  cxx_flag: "-cl-mad-enable"
  cxx_flag: "-sycl-spirv"
  cxx_flag: "-DTENSORFLOW_USE_SYCL=1"
  cxx_flag: "-DEIGEN_USE_SYCL=1"
  cxx_flag: "-DEIGEN_HAS_C99_MATH=1"
  cxx_flag: "-DEIGEN_HAS_CXX11_MATH=1"

  # Make C++ compilation deterministic. Use linkstamping instead of these
  # compiler symbols.
  unfiltered_cxx_flag: "-Wno-builtin-macro-redefined"
  unfiltered_cxx_flag: "-D__DATE__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIMESTAMP__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIME__=\"redacted\""

  compiler_flag: "-fPIE"

  # Keep stack frames for debugging, even in opt mode.
  compiler_flag: "-fno-omit-frame-pointer"

  # Anticipated future default.
  linker_flag: "-no-canonical-prefixes"

  # All warnings are enabled. Maybe enable -Werror as well?
  compiler_flag: "-Wall"

  # Anticipated future default.
  linker_flag: "-Wl,-no-as-needed"
  # Stamp the binary with a unique identifier.
  linker_flag: "-Wl,--build-id=md5"
  linker_flag: "-Wl,--hash-style=gnu"

  linking_mode_flags { mode: DYNAMIC }

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
  target_libc: "armeabi"
  target_cpu: "armeabi"
  target_system_name: "armeabi"
  toolchain_identifier: "arm-linux-gnueabihf"

  tool_path { name: "ar" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-ar" }
  tool_path { name: "compat-ld" path: "/bin/false" }
  tool_path { name: "cpp" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-cpp" }
  tool_path { name: "dwp" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-dwp" }
  tool_path { name: "gcc" path: "%{COMPUTECPP_ROOT_DIR}%/bin/compute++" }
  tool_path { name: "gcov" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-gcov" }
  tool_path { name: "ld" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-ld" }
  tool_path { name: "nm" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-nm" }
  tool_path { name: "objcopy" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-objcopy" }
  tool_path { name: "objdump" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-objdump" }
  tool_path { name: "strip" path: "%{ARM_COMPILER_PATH}%/bin/arm-linux-gnueabihf-strip" }

  cxx_builtin_include_directory: "%{ARM_COMPILER_PATH}%/arm-linux-gnueabihf/include/c++/%{VERSION}%"
  cxx_builtin_include_directory: "%{ARM_COMPILER_PATH}%/arm-linux-gnueabihf/sysroot/usr/include/"
  cxx_builtin_include_directory: "%{ARM_COMPILER_PATH}%/arm-linux-gnueabihf/libc/usr/include/"
  cxx_builtin_include_directory: "%{ARM_COMPILER_PATH}%/lib/gcc/arm-linux-gnueabihf/%{VERSION}%/include"
  cxx_builtin_include_directory: "%{ARM_COMPILER_PATH}%/lib/gcc/arm-linux-gnueabihf/%{VERSION}%/include-fixed"
  cxx_builtin_include_directory: "%{ARM_COMPILER_PATH}%/local_include"
  cxx_builtin_include_directory: "/usr/include"
  cxx_flag: "-std=c++11"
  cxx_flag: "-isystem"
  cxx_flag: "%{PYTHON_INCLUDE_PATH}%"
  cxx_flag: "-target arm-linux-gnueabihf"
  cxx_flag: "-fsycl-ih-last"
  cxx_flag: "-sycl-driver"
  cxx_flag: "-no-serial-memop"
  cxx_flag: "-Xclang"
  cxx_flag: "-cl-denorms-are-zero"
  cxx_flag: "-Xclang"
  cxx_flag: "-cl-fp32-correctly-rounded-divide-sqrt"
  cxx_flag: "-Xclang"
  cxx_flag: "-cl-mad-enable"
  cxx_flag: "-sycl-spirv"
  cxx_flag: "-DTENSORFLOW_USE_SYCL=1"
  cxx_flag: "-DEIGEN_USE_SYCL=1"
  cxx_flag: "-DEIGEN_HAS_C99_MATH=1"
  cxx_flag: "-DEIGEN_HAS_CXX11_MATH=1"
  linker_flag: "-lstdc++"

  unfiltered_cxx_flag: "-Wno-builtin-macro-redefined"
  unfiltered_cxx_flag: "-D__DATE__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIMESTAMP__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIME__=\"redacted\""
  unfiltered_cxx_flag: "-no-canonical-prefixes"
  unfiltered_cxx_flag: "-fno-canonical-system-headers"

  compiler_flag: "-U_FORTIFY_SOURCE"
  compiler_flag: "-D_FORTIFY_SOURCE=1"
  compiler_flag: "-fstack-protector"

  linker_flag: "-Wl,-z,relro,-z,now"
  linker_flag: "-no-canonical-prefixes"
  linker_flag: "-pass-exit-codes"
  linker_flag: "-Wl,--build-id=md5"
  linker_flag: "-Wl,--hash-style=gnu"

  compilation_mode_flags {
    mode: DBG
    # Enable debug symbols.
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
