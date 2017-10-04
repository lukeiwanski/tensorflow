licenses(["notice"]) # MIT License

exports_files(["LICENSE"])

config_setting(
    name = "using_acl",
    values = {
        "define": "using_acl=true",
    },
    visibility = ["//visibility:public"],
)

load(
    "//third_party/acl:build_defs.bzl",
    "if_acl",
)

cc_library(
    name = "intel_binary_blob",
    srcs = if_acl([
        "@acl//:libarm_compute.so",
        "@acl//:libOpenCL.so",
    ]),
    visibility = ["//visibility:public"],
    deps = ["@acl//:acl_headers"],
)
