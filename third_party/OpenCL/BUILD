licenses(["notice"]) # KHR licence

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "OpenCL",
    srcs = [
        "icd.c",
        "icd_dispatch.c",
        "icd_linux.c",
    ],
    deps = [
        "@khronos_opencl_headers//:OpenCL-Headers",
    ],
)
