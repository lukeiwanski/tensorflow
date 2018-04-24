licenses(["notice"])  # Apache 2.0

load("@local_config_sycl//sycl:build_defs.bzl", "if_sycl")
load("@local_config_sycl//sycl:platform.bzl", "sycl_library_path")

load(":platform.bzl", "readlink_command")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE.text"])

config_setting(
    name = "using_sycl_ccpp",
    define_values = {
        "using_sycl": "true",
        "using_trisycl": "false",
    },
)

config_setting(
    name = "using_sycl_trisycl",
    define_values = {
        "using_sycl": "true",
        "using_trisycl": "true",
    },
)


cc_library(
    name = "sycl_headers",
    hdrs = glob([
        "**/*.h",
        "**/*.hpp",
    ]) + ["@opencl_headers//:OpenCL-Headers"],
    includes = [".", "include"],
    deps = ["@opencl_headers//:OpenCL-Headers"],
)

cc_library(
    name = "syclrt",
    srcs = [
        sycl_library_path("ComputeCpp")
    ],
    data = [
        sycl_library_path("ComputeCpp")
    ],
    includes = ["include/"],
)

cc_library(
    name = "sycl",
    deps = if_sycl([
        ":sycl_headers",
        ":syclrt",
    ]),
)
