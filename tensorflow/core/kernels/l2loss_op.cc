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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/l2loss_op.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

template <typename T>
class L2LossOp<CPUDevice, T> : public OpKernel {
 public:
  explicit L2LossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // The input tensor can be of any number of dimensions, even though it's
    // 2D in most typical applications.
    const Tensor& input = context->input(0);
    // The output is a single number.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    const CPUDevice& d = context->eigen_device<CPUDevice>();
    output->scalar<T>().device(d) =
        (input.flat<T>().square() * static_cast<T>(0.5)).sum();
  }
};

#define REGISTER_KERNEL(T)                                      \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("L2Loss").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      L2LossOp<CPUDevice, T>);

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
REGISTER_KERNEL(Eigen::half);
#undef REGISTER_KERNEL


#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("L2Loss").Device(DEVICE_SYCL).TypeConstraint<T>("T"), \
      L2LossOp<SYCLDevice, T>);

REGISTER_SYCL_KERNEL(float);
REGISTER_SYCL_KERNEL(double);
#undef REGISTER_SYCL_KERNEL
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
