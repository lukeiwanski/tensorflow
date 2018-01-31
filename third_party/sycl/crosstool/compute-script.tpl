#!/bin/bash

set -eu

exec %{COMPUTECPP_ROOT_DIR}%/bin/compute -target %{ARM_TARGET}% \
    --gcc-toolchain=%{ARM_COMPILER_PATH}% \
    --sysroot %{ARM_COMPILER_PATH}%/%{ARM_TARGET}%/libc \
    "$@"
