/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
REGISTER6(BinaryOp, CPU, "NotEqual", functor::not_equal_to, float, Eigen::half,
          double, uint8, int8, int16);
#if GOOGLE_CUDA
REGISTER4(BinaryOp, GPU, "NotEqual", functor::not_equal_to, float, Eigen::half,
          double, uint8);
#endif

#ifdef TENSORFLOW_USE_SYCL
REGISTER(BinaryOp, SYCL, "NotEqual", functor::not_equal_to, uint8);
#define REGISTER_SYCL(type) \
  REGISTER(BinaryOp, SYCL, "NotEqual", functor::not_equal_to, type)
TF_CALL_SYCL_NUMBER_TYPES(REGISTER_SYCL);
#undef REGISTER_SYCL
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
