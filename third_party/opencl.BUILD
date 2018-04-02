licenses(["notice"]) # KHR licence

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "OpenCL",
    srcs = [
        "icd.c",
        "icd_dispatch.c",
        "icd_linux.c",
    ],
    hdrs = [
        "@opencl_headers//:OpenCL-Headers",
    ],
    deps = [
        "@opencl_headers//:OpenCL-Headers",
        ":icd_exports.ldscript",
    ],
    includes = [
        ".",
    ],
    linkopts = [
        "-Wl,--version-script",
        ":icd_exports.ldscript",
    ],
)
