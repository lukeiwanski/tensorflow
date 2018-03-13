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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include <cfloat>
#include <vector>

#include "tensorflow/core/kernels/dilation_ops.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

void ParseAttributes(OpKernelConstruction* context, std::vector<int32>* strides,
                     std::vector<int32>* rates, Padding* padding) {
  OP_REQUIRES_OK(context, context->GetAttr("strides", strides));
  OP_REQUIRES(context, strides->size() == 4,
              errors::InvalidArgument("Sliding window stride field must "
                                      "specify 4 dimensions"));
  OP_REQUIRES(context, (*strides)[0] == 1 && (*strides)[3] == 1,
              errors::Unimplemented(
                  "Stride is only supported across spatial dimensions."));

  OP_REQUIRES_OK(context, context->GetAttr("rates", rates));
  OP_REQUIRES(context, rates->size() == 4,
              errors::InvalidArgument("Input stride (atrous rate) field "
                                      "must specify 4 dimensions"));
  OP_REQUIRES(context, (*rates)[0] == 1 && (*rates)[3] == 1,
              errors::Unimplemented(
                  "Rate is only supported across spatial dimensions."));

  OP_REQUIRES_OK(context, context->GetAttr("padding", padding));
}

void ParseSizes(OpKernelContext* context, const std::vector<int32>& strides,
                const std::vector<int32>& rates, const Padding& padding,
                int* stride_rows, int* stride_cols, int* rate_rows,
                int* rate_cols, int64* pad_top, int64* pad_left,
                int64* out_rows, int64* out_cols) {
  // Input tensor is of the following dimensions:
  // [ batch, input_rows, input_cols, depth ]
  const Tensor& input = context->input(0);
  OP_REQUIRES(context, input.dims() == 4,
              errors::InvalidArgument("input must be 4-dimensional",
                                      input.shape().DebugString()));
  const int input_rows = input.dim_size(1);
  const int input_cols = input.dim_size(2);
  const int depth = input.dim_size(3);

  // For now we take the stride and rate from the second and third dimensions
  // only (we do not support striding on the batch or depth dimension).
  *stride_rows = strides[1];
  *stride_cols = strides[2];
  *rate_rows = rates[1];
  *rate_cols = rates[2];

  // Input filter is of the following dimensions:
  // [ filter_rows, filter_cols, depth ]
  const Tensor& filter = context->input(1);
  OP_REQUIRES(context, filter.dims() == 3,
              errors::InvalidArgument("filter must be 3-dimensional: ",
                                      filter.shape().DebugString()));
  const int filter_rows = filter.dim_size(0);
  const int filter_cols = filter.dim_size(1);
  OP_REQUIRES(
      context, depth == filter.dim_size(2),
      errors::InvalidArgument("input and filter must have the same depth: ",
                              depth, " vs ", filter.dim_size(2)));

  // Effective filter size, after introducing rate - 1 zeros between each
  // non-zero filter element.
  const int filter_rows_eff =
      filter_rows + (filter_rows - 1) * (*rate_rows - 1);
  const int filter_cols_eff =
      filter_cols + (filter_cols - 1) * (*rate_cols - 1);

  OP_REQUIRES_OK(
      context, GetWindowedOutputSize(input_rows, filter_rows_eff, *stride_rows,
                                     padding, out_rows, pad_top));
  OP_REQUIRES_OK(
      context, GetWindowedOutputSize(input_cols, filter_cols_eff, *stride_cols,
                                     padding, out_cols, pad_left));
}

template <typename Device, typename T>
class DilationOp : public OpKernel {
 public:
  explicit DilationOp(OpKernelConstruction* context) : OpKernel(context) {
    ParseAttributes(context, &strides_, &rates_, &padding_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);

    // Determine relevant sizes from input and filters.
    int stride_rows = 0, stride_cols = 0;
    int rate_rows = 0, rate_cols = 0;
    int64 pad_top = 0, pad_left = 0;
    int64 out_rows = 0, out_cols = 0;
    ParseSizes(context, strides_, rates_, padding_, &stride_rows, &stride_cols,
               &rate_rows, &rate_cols, &pad_top, &pad_left, &out_rows,
               &out_cols);

    // Output tensor is of the following dimensions:
    // [ batch, out_rows, out_cols, depth ]
    const int batch = input.dim_size(0);
    const int depth = input.dim_size(3);
    const std::vector<int64> out_sizes = {batch, out_rows, out_cols, depth};
    TensorShape out_shape(out_sizes);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    functor::Dilation<Device, T>()(
        context->eigen_device<Device>(), input.tensor<T, 4>(),
        filter.tensor<T, 3>(), stride_rows, stride_cols, rate_rows, rate_cols,
        pad_top, pad_left, output->tensor<T, 4>());
  }

  std::vector<int32> strides_;
  std::vector<int32> rates_;
  Padding padding_;
};

// Partial specialization of Dilation functor for a CPUDevice.
namespace functor {
template <typename T>
struct Dilation<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter, int stride_rows,
                  int stride_cols, int rate_rows, int rate_cols, int pad_top,
                  int pad_left, typename TTypes<T, 4>::Tensor output) {
    const int batch = input.dimension(0);
    const int input_rows = input.dimension(1);
    const int input_cols = input.dimension(2);
    const int depth = input.dimension(3);

    const int filter_rows = filter.dimension(0);
    const int filter_cols = filter.dimension(1);

    const int output_rows = output.dimension(1);
    const int output_cols = output.dimension(2);

    // This is a reference implementation, likely to be slow.
    // TODO(gpapan): Write multi-threaded implementation.
    for (int b = 0; b < batch; ++b) {
      for (int h_out = 0; h_out < output_rows; ++h_out) {
        int h_beg = h_out * stride_rows - pad_top;
        for (int w_out = 0; w_out < output_cols; ++w_out) {
          int w_beg = w_out * stride_cols - pad_left;
          for (int d = 0; d < depth; ++d) {
            T cur_val = Eigen::NumTraits<T>::lowest();
            for (int h = 0; h < filter_rows; ++h) {
              const int h_in = h_beg + h * rate_rows;
              if (h_in >= 0 && h_in < input_rows) {
                for (int w = 0; w < filter_cols; ++w) {
                  const int w_in = w_beg + w * rate_cols;
                  if (w_in >= 0 && w_in < input_cols) {
                    const T val = input(b, h_in, w_in, d) + filter(h, w, d);
                    if (val > cur_val) {
                      cur_val = val;
                    }
                  }
                }
              }
            }
            output(b, h_out, w_out, d) = cur_val;
          }
        }
      }
    }
  }
};
}  // namespace functor

#ifdef TENSORFLOW_USE_SYCL
template <typename T>
class Dilation2DSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  Dilation2DSYCL(int batch, int input_rows, int input_cols,
                 int depth, int filter_rows, int filter_cols,
                 int output_rows, int output_cols, int stride_rows,
                 int stride_cols, int rate_rows, int rate_cols,
                 int pad_top, int pad_left,
                 const read_accessor input_accessor,
                 const read_accessor filter_accessor,
                 write_accessor output_accessor)
      : batch_(batch),
        input_rows_(input_rows),
        input_cols_(input_cols),
        depth_(depth),
        filter_rows_(filter_rows),
        filter_cols_(filter_cols),
        output_rows_(output_rows),
        output_cols_(output_cols),
        stride_rows_(stride_rows),
        stride_cols_(stride_cols),
        rate_rows_(rate_rows),
        rate_cols_(rate_cols),
        pad_top_(pad_top),
        pad_left_(pad_left),
        input_accessor_(input_accessor),
        filter_accessor_(filter_accessor),
        output_accessor_(output_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
    T* filter_data = ConvertToActualTypeSycl(T, filter_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

    int out_idx = item.get_linear_id();
    const int d = out_idx % depth_;
    const int out_idx2 = out_idx / depth_;
    const int w_out = out_idx2 % output_cols_;
    const int out_idx3 = out_idx2 / output_cols_;
    const int h_out = out_idx3 % output_rows_;
    const int b = out_idx3 / output_rows_;
    int h_beg = h_out * stride_rows_ - pad_top_;
    int w_beg = w_out * stride_cols_ - pad_left_;
    T cur_val = Eigen::NumTraits<T>::lowest();
    for (int h = 0; h < filter_rows_; ++h) {
      const int h_in = h_beg + h * rate_rows_;
      if (h_in >= 0 && h_in < input_rows_) {
        for (int w = 0; w < filter_cols_; ++w) {
          const int w_in = w_beg + w * rate_cols_;
          if (w_in >= 0 && w_in < input_cols_) {
            const T val =
                input_data[d +
                          depth_ *
                              (w_in + input_cols_ * (h_in + input_rows_ * b))] +
                filter_data[d + depth_ * (w + filter_cols_ * h)];
            if (val > cur_val) {
              cur_val = val;
            }
          }
        }
      }
    }
    output_data[out_idx] = cur_val;
  }

 private:

   int batch_;
   int input_rows_;
   int input_cols_;
   int depth_;
   int filter_rows_;
   int filter_cols_;
   int output_rows_;
   int output_cols_;
   int stride_rows_;
   int stride_cols_;
   int rate_rows_;
   int rate_cols_;
   int pad_top_;
   int pad_left_;
  const read_accessor input_accessor_;
  const read_accessor filter_accessor_;
  write_accessor output_accessor_;
};

namespace functor {

template <typename T>
struct Dilation<SYCLDevice, T> {
  void operator()(const SYCLDevice& device, const Tensor& input,
                  const Tensor& filter, int stride_rows,
                  int stride_cols, int rate_rows, int rate_cols, int pad_top,
                  int pad_left, Tensor* output) {
    auto input_tensor = input.tensor<T, 4>();
    auto filter_tensor = filter.tensor<T, 3>();
    auto output_tensor = output->tensor<T, 4>();

    const int batch = input_tensor.dimension(0);
    const int input_rows = input_tensor.dimension(1);
    const int input_cols = input_tensor.dimension(2);
    const int depth = input_tensor.dimension(3);

    const int filter_rows = filter_tensor.dimension(0);
    const int filter_cols = filter_tensor.dimension(1);

    const int output_rows = output_tensor.dimension(1);
    const int output_cols = output_tensor.dimension(2);

    const int num_threads = output->NumElements();

    auto input_buffer =
        device.get_sycl_buffer(input.template flat<T>().data());
    auto filter_buffer =
        device.get_sycl_buffer(filter.template flat<T>().data());
    auto output_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access =
          input_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
      auto filter_access =
          filter_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_access =
          output_buffer.template get_access<cl::sycl::access::mode::write>(cgh);
      Dilation2DSYCL<T> dilation(batch, input_rows, input_cols, depth,
                                 filter_rows, filter_cols, output_rows,
                                 output_cols, stride_rows, stride_cols,
                                 rate_rows, rate_cols, pad_top, pad_left,
                                 input_access, filter_access, output_access);

      cgh.parallel_for(cl::sycl::range<1>(num_threads), dilation);
    });
  }
};
}  // namespace functor

template <typename T>
class DilationOp<SYCLDevice, T> : public OpKernel {
 public:
  explicit DilationOp(OpKernelConstruction* context) : OpKernel(context) {
    ParseAttributes(context, &strides_, &rates_, &padding_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);

    // Determine relevant sizes from input and filters.
    int stride_rows = 0, stride_cols = 0;
    int rate_rows = 0, rate_cols = 0;
    int64 pad_top = 0, pad_left = 0;
    int64 out_rows = 0, out_cols = 0;
    ParseSizes(context, strides_, rates_, padding_, &stride_rows, &stride_cols,
               &rate_rows, &rate_cols, &pad_top, &pad_left, &out_rows,
               &out_cols);

    // Output tensor is of the following dimensions:
    // [ batch, out_rows, out_cols, depth ]
    const int batch = input.dim_size(0);
    const int depth = input.dim_size(3);
    const std::vector<int64> out_sizes = {batch, out_rows, out_cols, depth};
    TensorShape out_shape(out_sizes);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    functor::Dilation<SYCLDevice, T>()(
        context->eigen_device<SYCLDevice>(), input,
        filter, stride_rows, stride_cols, rate_rows, rate_cols,
        pad_top, pad_left, output);
  }

  std::vector<int32> strides_;
  std::vector<int32> rates_;
  Padding padding_;
};
#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename T>
class DilationBackpropInputOp : public OpKernel {
 public:
  explicit DilationBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    ParseAttributes(context, &strides_, &rates_, &padding_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // Determine relevant sizes from input and filters.
    int stride_rows = 0, stride_cols = 0;
    int rate_rows = 0, rate_cols = 0;
    int64 pad_top = 0, pad_left = 0;
    int64 out_rows = 0, out_cols = 0;
    ParseSizes(context, strides_, rates_, padding_, &stride_rows, &stride_cols,
               &rate_rows, &rate_cols, &pad_top, &pad_left, &out_rows,
               &out_cols);

    // Verify that the incoming gradient tensor has the expected size
    // [ batch, out_rows, out_cols, depth ]
    const int batch = input.dim_size(0);
    const int depth = input.dim_size(3);
    OP_REQUIRES(context, batch == out_backprop.dim_size(0) &&
                             out_rows == out_backprop.dim_size(1) &&
                             out_cols == out_backprop.dim_size(2) &&
                             depth == out_backprop.dim_size(3),
                errors::InvalidArgument("out_backprop has incompatible size."));

    // The computed in_backprop has the same dimensions as the input:
    // [ batch, input_rows, input_cols, depth ]
    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &in_backprop));

    // If there is nothing to compute, return.
    if (input.shape().num_elements() == 0) {
      return;
    }

    functor::DilationBackpropInput<Device, T>()(
        context->eigen_device<Device>(), input.tensor<T, 4>(),
        filter.tensor<T, 3>(), out_backprop.tensor<T, 4>(), stride_rows,
        stride_cols, rate_rows, rate_cols, pad_top, pad_left,
        in_backprop->tensor<T, 4>());
  }

  std::vector<int32> strides_;
  std::vector<int32> rates_;
  Padding padding_;
};

// Partial specialization of DilationBackpropInput functor for a CPUDevice.
namespace functor {
template <typename T>
struct DilationBackpropInput<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor out_backprop,
                  int stride_rows, int stride_cols, int rate_rows,
                  int rate_cols, int pad_top, int pad_left,
                  typename TTypes<T, 4>::Tensor in_backprop) {
    const int batch = input.dimension(0);
    const int input_rows = input.dimension(1);
    const int input_cols = input.dimension(2);
    const int depth = input.dimension(3);

    const int filter_rows = filter.dimension(0);
    const int filter_cols = filter.dimension(1);

    const int output_rows = out_backprop.dimension(1);
    const int output_cols = out_backprop.dimension(2);

    // Initialize gradient with all zeros.
    in_backprop.setZero();

    // This is a reference implementation, likely to be slow.
    // TODO(gpapan): Write multi-threaded implementation.
    // In the case of multiple argmax branches, we only back-propagate along the
    // last branch, i.e., the one with largest value of `h * filter_cols + w`,
    // similarly to the max-pooling backward routines.
    for (int b = 0; b < batch; ++b) {
      for (int h_out = 0; h_out < output_rows; ++h_out) {
        int h_beg = h_out * stride_rows - pad_top;
        for (int w_out = 0; w_out < output_cols; ++w_out) {
          int w_beg = w_out * stride_cols - pad_left;
          for (int d = 0; d < depth; ++d) {
            T cur_val = Eigen::NumTraits<T>::lowest();
            int h_in_max = (h_beg < 0) ? 0 : h_beg;
            int w_in_max = (w_beg < 0) ? 0 : w_beg;
            for (int h = 0; h < filter_rows; ++h) {
              const int h_in = h_beg + h * rate_rows;
              if (h_in >= 0 && h_in < input_rows) {
                for (int w = 0; w < filter_cols; ++w) {
                  const int w_in = w_beg + w * rate_cols;
                  if (w_in >= 0 && w_in < input_cols) {
                    const T val = input(b, h_in, w_in, d) + filter(h, w, d);
                    if (val > cur_val) {
                      cur_val = val;
                      h_in_max = h_in;
                      w_in_max = w_in;
                    }
                  }
                }
              }
            }
            in_backprop(b, h_in_max, w_in_max, d) +=
                out_backprop(b, h_out, w_out, d);
          }
        }
      }
    }
  }
};
}  // namespace functor

template <typename Device, typename T>
class DilationBackpropFilterOp : public OpKernel {
 public:
  explicit DilationBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
    ParseAttributes(context, &strides_, &rates_, &padding_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // Determine relevant sizes from input and filters.
    int stride_rows = 0, stride_cols = 0;
    int rate_rows = 0, rate_cols = 0;
    int64 pad_top = 0, pad_left = 0;
    int64 out_rows = 0, out_cols = 0;
    ParseSizes(context, strides_, rates_, padding_, &stride_rows, &stride_cols,
               &rate_rows, &rate_cols, &pad_top, &pad_left, &out_rows,
               &out_cols);

    // Verify that the incoming gradient tensor has the expected size
    // [ batch, out_rows, out_cols, depth ]
    const int batch = input.dim_size(0);
    const int depth = input.dim_size(3);
    OP_REQUIRES(context, batch == out_backprop.dim_size(0) &&
                             out_rows == out_backprop.dim_size(1) &&
                             out_cols == out_backprop.dim_size(2) &&
                             depth == out_backprop.dim_size(3),
                errors::InvalidArgument("out_backprop has incompatible size."));

    // The computed filter_backprop has the same dimensions as the filter:
    // [ batch, input_rows, input_cols, depth ]
    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, filter.shape(), &filter_backprop));

    // If there is nothing to compute, return.
    if (filter.shape().num_elements() == 0) {
      return;
    }

    functor::DilationBackpropFilter<Device, T>()(
        context->eigen_device<Device>(), input.tensor<T, 4>(),
        filter.tensor<T, 3>(), out_backprop.tensor<T, 4>(), stride_rows,
        stride_cols, rate_rows, rate_cols, pad_top, pad_left,
        filter_backprop->tensor<T, 3>());
  }

  std::vector<int32> strides_;
  std::vector<int32> rates_;
  Padding padding_;
};

// Partial specialization of DilationBackpropFilter functor for a CPUDevice.
namespace functor {
template <typename T>
struct DilationBackpropFilter<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor out_backprop,
                  int stride_rows, int stride_cols, int rate_rows,
                  int rate_cols, int pad_top, int pad_left,
                  typename TTypes<T, 3>::Tensor filter_backprop) {
    const int batch = input.dimension(0);
    const int input_rows = input.dimension(1);
    const int input_cols = input.dimension(2);
    const int depth = input.dimension(3);

    const int filter_rows = filter.dimension(0);
    const int filter_cols = filter.dimension(1);

    const int output_rows = out_backprop.dimension(1);
    const int output_cols = out_backprop.dimension(2);

    // Initialize gradient with all zeros.
    filter_backprop.setZero();

    // This is a reference implementation, likely to be slow.
    // TODO(gpapan): Write multi-threaded implementation.
    // In the case of multiple argmax branches, we only back-propagate along the
    // last branch, i.e., the one with largest value of `h * filter_cols + w`,
    // similarly to the max-pooling backward routines.
    for (int b = 0; b < batch; ++b) {
      for (int h_out = 0; h_out < output_rows; ++h_out) {
        int h_beg = h_out * stride_rows - pad_top;
        for (int w_out = 0; w_out < output_cols; ++w_out) {
          int w_beg = w_out * stride_cols - pad_left;
          for (int d = 0; d < depth; ++d) {
            T cur_val = Eigen::NumTraits<T>::lowest();
            int h_max = 0;
            int w_max = 0;
            for (int h = 0; h < filter_rows; ++h) {
              const int h_in = h_beg + h * rate_rows;
              if (h_in >= 0 && h_in < input_rows) {
                for (int w = 0; w < filter_cols; ++w) {
                  const int w_in = w_beg + w * rate_cols;
                  if (w_in >= 0 && w_in < input_cols) {
                    const T val = input(b, h_in, w_in, d) + filter(h, w, d);
                    if (val > cur_val) {
                      cur_val = val;
                      h_max = h;
                      w_max = w;
                    }
                  }
                }
              }
            }
            filter_backprop(h_max, w_max, d) +=
                out_backprop(b, h_out, w_out, d);
          }
        }
      }
    }
  }
};
}  // namespace functor

#define REGISTER(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Dilation2D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DilationOp<CPUDevice, T>);                                    \
                                                                    \
  REGISTER_KERNEL_BUILDER(Name("Dilation2DBackpropInput")           \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T"),              \
                          DilationBackpropInputOp<CPUDevice, T>);   \
                                                                    \
  REGISTER_KERNEL_BUILDER(Name("Dilation2DBackpropFilter")          \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T"),              \
                          DilationBackpropFilterOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA

#define REGISTER(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Dilation2D").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DilationOp<GPUDevice, T>);                                    \
                                                                    \
  REGISTER_KERNEL_BUILDER(Name("Dilation2DBackpropInput")           \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T"),              \
                          DilationBackpropInputOp<GPUDevice, T>);   \
                                                                    \
  REGISTER_KERNEL_BUILDER(Name("Dilation2DBackpropFilter")          \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T"),              \
                          DilationBackpropFilterOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL

#define REGISTER(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Dilation2D").Device(DEVICE_SYCL).TypeConstraint<T>("T"), \
      DilationOp<SYCLDevice, T>);                                     \

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER);

#undef REGISTER

#endif

}  // namespace tensorflow
