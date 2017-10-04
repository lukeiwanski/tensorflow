licenses(["notice"])  # MIT

exports_files(["license.txt"])

filegroup(
    name = "LICENSE",
    srcs = [
        "license.txt",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "acl_headers",
    srcs = glob(["**/*.h"]),
    includes = [".", "include", "arm_compute", "support", "utils"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "libarm_compute.so",
    srcs = ["lib/libarm_compute.so"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "libOpenCL.so",
    srcs = ["lib/libOpenCL.so"],
    visibility = ["//visibility:public"],
)
