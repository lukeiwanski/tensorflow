/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER5(UnaryOp, CPU, "Log1p", functor::log1p, float, Eigen::half, double,
          complex64, complex128);

#if GOOGLE_CUDA
REGISTER3(UnaryOp, GPU, "Log1p", functor::log1p, float, Eigen::half, double);
#endif

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_CWISE_KERNEL(type)                          \
REGISTER_KERNEL_BUILDER(Name("Log1p")                             \
                            .Device(DEVICE_SYCL)                  \
                            .TypeConstraint<type>("T"),           \
                        UnaryOp<SYCLDevice, functor::log1p<type>>);
TF_CALL_SYCL_NUMBER_TYPES(REGISTER_SYCL_CWISE_KERNEL);
#undef REGISTER_SYCL_CWISE_KERNEL
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
