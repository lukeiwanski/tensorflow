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

#include "tensorflow/core/kernels/maxpooling_op.h"

#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/eigen_pooling.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/maxpooling_op_gpu.h"
#include "tensorflow/core/kernels/pooling_ops_common_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

const int kInvalidMaxPoolingIndex = -1;

template <typename Device, typename T>
static void SpatialMaxPoolWithArgMaxHelper(
    OpKernelContext* context, Tensor* output, Tensor* output_arg_max,
    Tensor* input_backprop, const Tensor& tensor_in, const Tensor& out_backprop,
    const PoolParameters& params, const Padding& padding) {
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      ConstEigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      EigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<int64, Eigen::Dynamic, Eigen::Dynamic>>
      EigenIndexMatrixMap;

  ConstEigenMatrixMap in_mat(
      tensor_in.flat<T>().data(), params.depth,
      params.tensor_in_cols * params.tensor_in_rows * params.tensor_in_batch);
  EigenMatrixMap out_mat(
      output->flat<T>().data(), params.depth,
      params.out_width * params.out_height * params.tensor_in_batch);
  EigenIndexMatrixMap out_arg_max_mat(
      output_arg_max->flat<int64>().data(), params.depth,
      params.out_width * params.out_height * params.tensor_in_batch);

  const DeviceBase::CpuWorkerThreads& worker_threads =
      *(context->device()->tensorflow_cpu_worker_threads());

  // The following code basically does the following:
  // 1. Flattens the input and output tensors into two dimensional arrays.
  //    tensor_in_as_matrix:
  //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
  //    output_as_matrix:
  //      depth by (out_width * out_height * tensor_in_batch)
  //
  // 2. Walks through the set of columns in the flattened tensor_in_as_matrix,
  //    and updates the corresponding column(s) in output_as_matrix with the
  //    max value.
  auto shard = [&params, &in_mat, &out_mat, &out_arg_max_mat, &input_backprop,
                &output_arg_max, &out_backprop](int64 start, int64 limit) {

    const int32 depth = params.depth;
    const int32 in_rows = params.tensor_in_rows;
    const int32 in_cols = params.tensor_in_cols;
    const int32 pad_rows = params.pad_rows;
    const int32 pad_cols = params.pad_cols;
    const int32 window_rows = params.window_rows;
    const int32 window_cols = params.window_cols;
    const int32 row_stride = params.row_stride;
    const int32 col_stride = params.col_stride;
    const int32 out_height = params.out_height;
    const int32 out_width = params.out_width;

    {
      // Initializes the output tensor with MIN<T>.
      const int32 output_image_size = out_height * out_width * depth;
      EigenMatrixMap out_shard(out_mat.data() + start * output_image_size, 1,
                               (limit - start) * output_image_size);
      out_shard.setConstant(Eigen::NumTraits<T>::lowest());
      EigenIndexMatrixMap out_arg_max_shard(
          out_arg_max_mat.data() + start * output_image_size, 1,
          (limit - start) * output_image_size);
      out_arg_max_shard.setConstant(kInvalidMaxPoolingIndex);
    }

    for (int64 b = start; b < limit; ++b) {
      for (int h = 0; h < in_rows; ++h) {
        for (int w = 0; w < in_cols; ++w) {
          // (h_start, h_end) * (w_start, w_end) is the range that the input
          // vector projects to.
          const int hpad = h + pad_rows;
          const int wpad = w + pad_cols;
          const int h_start =
              (hpad < window_rows) ? 0 : (hpad - window_rows) / row_stride + 1;
          const int h_end = std::min(hpad / row_stride + 1, out_height);
          const int w_start =
              (wpad < window_cols) ? 0 : (wpad - window_cols) / col_stride + 1;
          const int w_end = std::min(wpad / col_stride + 1, out_width);
          // compute elementwise max
          const int64 in_index = (b * in_rows + h) * in_cols + w;
          for (int ph = h_start; ph < h_end; ++ph) {
            const int64 out_index_base = (b * out_height + ph) * out_width;
            for (int pw = w_start; pw < w_end; ++pw) {
              const int64 out_index = out_index_base + pw;
              /// NOTES(zhengxq): not using the eigen matrix operation for
              /// now.
              for (int d = 0; d < depth; ++d) {
                const T& input_ref = in_mat.coeffRef(d, in_index);
                T& output_ref = out_mat.coeffRef(d, out_index);
                int64& out_arg_max_ref = out_arg_max_mat.coeffRef(d, out_index);
                if (output_ref < input_ref ||
                    out_arg_max_ref == kInvalidMaxPoolingIndex) {
                  output_ref = input_ref;
                  int64 input_offset = in_index * depth + d;
                  out_arg_max_ref = input_offset;
                }
              }
            }
          }
        }
      }
    }

    {
      auto input_backprop_flat = input_backprop->flat<T>();
      auto out_arg_max_flat = output_arg_max->flat<int64>();
      auto out_backprop_flat = out_backprop.flat<T>();

      // Initialize output to 0.
      const int64 in_size = in_rows * in_cols * depth;
      const int64 in_start = start * in_size;
      const int64 in_end = limit * in_size;
      EigenMatrixMap in_shard(input_backprop_flat.data() + in_start, 1,
                              in_end - in_start);
      in_shard.setConstant(T(0));

      // Backpropagate.
      const int out_size = out_height * out_width * depth;
      const int out_start = start * out_size;
      const int out_end = limit * out_size;
      for (int index = out_start; index < out_end; ++index) {
        int input_backprop_index = out_arg_max_flat(index);
        // Although this check is in the inner loop, it is worth its value
        // so we don't end up with memory corruptions. Our benchmark shows that
        // the performance impact is quite small
        CHECK(input_backprop_index >= in_start && input_backprop_index < in_end)
            << "Invalid input backprop index: " << input_backprop_index << ", "
            << in_start << ", " << in_end;
        input_backprop_flat(input_backprop_index) += out_backprop_flat(index);
      }
    }

  };

  const int64 shard_cost = params.tensor_in_rows * params.tensor_in_cols *
                           params.depth * params.window_rows *
                           params.window_cols;
  Shard(worker_threads.num_threads, worker_threads.workers,
        params.tensor_in_batch, shard_cost, shard);
}

#ifdef TENSORFLOW_USE_SYCL
// MaxPool2D SYCL kernel. Expects the number of threads to be equal to the
// number of elements in the output tensor.
//
// For each output element, find the corresponding input window and run over
// all values in the window to find the maximum value. This value is then
// copied into that output element.
template <typename T>
class MaxPool2DSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  MaxPool2DSYCL(const PoolParameters& params,
                const read_accessor input_accessor,
                write_accessor output_accessor)
      : p_(params),
        input_accessor_(input_accessor),
        output_accessor_(output_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

    int index = item.get_linear_id();
    int n = index;
    int d = n % p_.depth_;
    n /= p_.depth_;
    int cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
    int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
    cstart = std::max(cstart, 0);
    n /= p_.out_cols_;
    int rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
    int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
    rstart = std::max(rstart, 0);
    n /= p_.out_rows_;
    T maxval = Eigen::NumTraits<T>::lowest();
    const T* input_data_n =
        input_data + n * p_.in_cols_ * p_.in_rows_ * p_.depth_;
    for (int r = rstart; r < rend; ++r) {
      for (int c = cstart; c < cend; ++c) {
        int idx = (r * p_.in_cols_ + c) * p_.depth_ + d;
        if (input_data_n[idx] > maxval) {
          maxval = input_data_n[idx];
        }
      }
    }
    output_data[index] = maxval;
  }

 private:
  const SYCL2DPoolParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};

template <typename T>
struct LaunchMaxPoolingOpSYCL {
  static void launch(OpKernelContext* context, Tensor* output,
                     const Tensor& tensor_in, const PoolParameters& params) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    const int num_threads = output->NumElements();

    auto input_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access =
          input_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_access =
          output_buffer.template get_access<cl::sycl::access::mode::write>(cgh);
      MaxPool2DSYCL<T> max_pool(params, input_access, output_access);

      cgh.parallel_for(cl::sycl::range<1>(num_threads), max_pool);
    });
  }
};

template <typename T>
class MaxPoolingOp<SYCLDevice, T> : public UnaryOp<T> {
 public:
  explicit MaxPoolingOp(OpKernelConstruction* context) : UnaryOp<T>(context) {
    string data_format;
    auto status = context->GetAttr("data_format", &data_format);
    if (status.ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument("Default MaxPoolingOp only supports NHWC."));
    } else {
      data_format_ = FORMAT_NHWC;
    }
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, params.forward_output_shape(), &output));
    LaunchMaxPoolingOpSYCL<T>::launch(context, output, tensor_in, params);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};
#endif  // TENSORFLOW_USE_SYCL

// The operation to compute MaxPool gradients.
// It takes three inputs:
//   - The original input tensor
//   - The original output tensor
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
template <class Device, class T>
class MaxPoolingGradOp : public OpKernel {
 public:
  explicit MaxPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("Default MaxPoolingGradOp only supports NHWC ",
                                "on device type ",
                                DeviceTypeString(context->device_type())));

    if (context->num_inputs() == 3) {
      OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
      OP_REQUIRES(context, ksize_.size() == 4,
                  errors::InvalidArgument("Sliding window ksize field must "
                                          "specify 4 dimensions"));
      OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
      OP_REQUIRES(context, stride_.size() == 4,
                  errors::InvalidArgument("Sliding window strides field must "
                                          "specify 4 dimensions"));
      OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                  errors::Unimplemented(
                      "Pooling is not yet supported on the batch dimension."));
      OP_REQUIRES(
          context, ksize_[3] == 1 && stride_[3] == 1,
          errors::Unimplemented(
              "MaxPoolingGrad is not yet supported on the depth dimension."));
    }

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // For maxpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    // For maxpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    const TensorShape& output_shape = tensor_in.shape();

    Tensor tensor_out_dup;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_temp(
                                {1}, DataTypeToEnum<T>::v(), tensor_out.shape(),
                                &tensor_out_dup));
    Tensor tensor_out_arg_max;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int64>::v(),
                                                   tensor_out.shape(),
                                                   &tensor_out_arg_max));
    std::vector<int32> ksize = ksize_;
    std::vector<int32> stride = stride_;
    if (context->num_inputs() == 5) {
      const Tensor& tensor_ksize = context->input(3);
      auto value_ksize = tensor_ksize.flat<int32>();
      ksize.resize(tensor_ksize.shape().num_elements());
      std::copy_n(&value_ksize(0), ksize.size(), ksize.begin());

      const Tensor& tensor_stride = context->input(4);
      auto value_stride = tensor_stride.flat<int32>();
      stride.resize(tensor_stride.shape().num_elements());
      std::copy_n(&value_stride(0), stride.size(), stride.begin());
    }

    OP_REQUIRES(context, ksize.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, stride.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, ksize[0] == 1 && stride[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(
        context, ksize[3] == 1 && stride[3] == 1,
        errors::Unimplemented(
            "MaxPoolingGrad is not yet supported on the depth dimension."));

    PoolParameters params{context,  ksize,       stride,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, output_shape, &output));

    SpatialMaxPoolWithArgMaxHelper<CPUDevice, T>(
        context, &tensor_out_dup, &tensor_out_arg_max, output, tensor_in,
        out_backprop, params, padding_);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

#ifdef GOOGLE_CUDA

template <typename T>
static void MaxPoolingBackwardCustomKernel(
    OpKernelContext* context, const std::vector<int32>& size,
    const std::vector<int32>& stride, Padding padding, const Tensor* tensor_in,
    const Tensor& out_backprop, const TensorShape& tensor_in_shape) {
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                              {0}, 0, tensor_in_shape, &output));

  PoolParameters params{context, size,        stride,
                        padding, FORMAT_NHWC, tensor_in_shape};
  if (!context->status().ok()) {
    return;
  }

  functor::MaxPoolBackwardNoMask<T>()(
      tensor_in->flat<T>().data(), params.tensor_in_batch,
      params.tensor_in_rows, params.tensor_in_cols, params.depth,
      params.out_height, params.out_width, params.window_rows,
      params.window_cols, params.row_stride, params.col_stride, params.pad_rows,
      params.pad_cols, out_backprop.flat<T>().data(), output->flat<T>().data(),
      context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class MaxPoolingGradOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit MaxPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    if (context->num_inputs() == 3) {
      OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
      OP_REQUIRES(context, ksize_.size() == 4,
                  errors::InvalidArgument("Sliding window ksize field must "
                                          "specify 4 dimensions"));
      OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
      OP_REQUIRES(context, stride_.size() == 4,
                  errors::InvalidArgument("Sliding window strides field must "
                                          "specify 4 dimensions"));
      const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
      const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
      OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                  errors::Unimplemented(
                      "Pooling is not yet supported on the batch dimension."));
    }
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));

    use_dnn_ = CanUseCudnn();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // For maxpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional 4"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    // For maxpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    TensorShape output_shape = tensor_in.shape();

    std::vector<int32> ksize = ksize_;
    std::vector<int32> stride = stride_;
    if (context->num_inputs() == 5) {
      const Tensor& tensor_ksize = context->input(3);
      auto value_ksize = tensor_ksize.flat<int32>();
      ksize.resize(tensor_ksize.shape().num_elements());
      std::copy_n(&value_ksize(0), ksize.size(), ksize.begin());

      const Tensor& tensor_stride = context->input(4);
      auto value_stride = tensor_stride.flat<int32>();
      stride.resize(tensor_stride.shape().num_elements());
      std::copy_n(&value_stride(0), stride.size(), stride.begin());
    }
    OP_REQUIRES(context, ksize.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, stride.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int32 ksize_n = GetTensorDim(ksize, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));

    if (use_dnn_) {
      DnnPoolingGradOp<T>::Compute(
          context, perftools::gputools::dnn::PoolingMode::kMaximum, ksize,
          stride, padding_, data_format_, &tensor_in, &tensor_out, out_backprop,
          output_shape);
    } else {
      CHECK(data_format_ == FORMAT_NHWC)
          << "Non-Cudnn MaxPoolGrad only supports NHWC format";
      MaxPoolingBackwardCustomKernel<T>(context, ksize, stride, padding_,
                                        &tensor_in, out_backprop, output_shape);
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool use_dnn_;
};

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
template <typename T>
struct LaunchMaxPoolingGradOpSYCL;

template <class T>
class MaxPoolingGradOp<SYCLDevice, T> : public OpKernel {
 public:
  explicit MaxPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, (GetTensorDim(ksize_, data_format_, 'N') == 1 &&
                          GetTensorDim(stride_, data_format_, 'N') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(context, (GetTensorDim(ksize_, data_format_, 'C') == 1 &&
                          GetTensorDim(stride_, data_format_, 'C') == 1),
                errors::Unimplemented(
                    "Pooling is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    const TensorShape& output_shape = tensor_in.shape();
    Tensor* input_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &input_backprop));
    std::array<int64, 2> input_size{
        {GetTensorDim(output_shape, data_format_, '1'),
         GetTensorDim(output_shape, data_format_, '0')}};
    std::array<int64, 2> window{{GetTensorDim(ksize_, data_format_, '1'),
                                 GetTensorDim(ksize_, data_format_, '0')}};
    std::array<int64, 2> stride{{GetTensorDim(stride_, data_format_, '1'),
                                 GetTensorDim(stride_, data_format_, '0')}};
    std::array<int64, 2> out, padding;

    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_size[0], window[0], stride[0],
                                         padding_, &out[0], &padding[0]));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_size[1], window[1], stride[1],
                                         padding_, &out[1], &padding[1]));
    LaunchMaxPoolingGradOpSYCL<T>::launch(
        context, tensor_in, tensor_out, out_backprop, window, stride, out,
        padding, data_format_, input_backprop);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};
#endif  // TENSORFLOW_USE_SYCL

// The operation to compute gradient of MaxPool gradients.
// It takes three inputs:
//   - The original input tensor
//   - The original output tensor
//   - Backprop tensor for output gradients
// It produces one output: backprop tensor for output gradient.
template <class Device, class T>
class MaxPoolingGradGradOp : public OpKernel {
 public:
  explicit MaxPoolingGradGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument(
            "Default MaxPoolingGradGradOp only supports NHWC ",
            "on device type ", DeviceTypeString(context->device_type())));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));

    if (context->num_inputs() == 3) {
      OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
      OP_REQUIRES(context, ksize_.size() == 4,
                  errors::InvalidArgument("Sliding window ksize field must "
                                          "specify 4 dimensions"));
      OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
      OP_REQUIRES(context, stride_.size() == 4,
                  errors::InvalidArgument("Sliding window strides field must "
                                          "specify 4 dimensions"));
      OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                  errors::Unimplemented(
                      "Pooling is not yet supported on the batch dimension."));
      OP_REQUIRES(context, ksize_[3] == 1 && stride_[3] == 1,
                  errors::Unimplemented("MaxPoolingGradGrad is not yet "
                                        "supported on the depth dimension."));
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_grad_backprop = context->input(2);

    // For maxpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    // For maxpooling, out_grad_backprop should have 4 dimensions.
    OP_REQUIRES(
        context, out_grad_backprop.dims() == 4,
        errors::InvalidArgument("out_grad_backprop must be 4-dimensional"));

    std::vector<int32> ksize = ksize_;
    std::vector<int32> stride = stride_;
    if (context->num_inputs() == 5) {
      const Tensor& tensor_ksize = context->input(3);
      auto value_ksize = tensor_ksize.flat<int32>();
      ksize.resize(tensor_ksize.shape().num_elements());
      std::copy_n(&value_ksize(0), ksize.size(), ksize.begin());

      const Tensor& tensor_stride = context->input(4);
      auto value_stride = tensor_stride.flat<int32>();
      stride.resize(tensor_stride.shape().num_elements());
      std::copy_n(&value_stride(0), stride.size(), stride.begin());
    }

    OP_REQUIRES(context, ksize.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, stride.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, ksize[0] == 1 && stride[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(
        context, ksize[3] == 1 && stride[3] == 1,
        errors::Unimplemented(
            "MaxPoolingGrad is not yet supported on the depth dimension."));

    PoolParameters params{context,  ksize,       stride,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {2}, 0, tensor_out.shape(), &output));

    SpatialMaxPoolGradGrad(context, output, tensor_in, tensor_out,
                           out_grad_backprop, params, padding_);
  }

 private:
  void SpatialMaxPoolGradGrad(OpKernelContext* context, Tensor* bottom_diff,
                              const Tensor& tensor_in, const Tensor& tensor_out,
                              const Tensor& top_diff,
                              const PoolParameters& params,
                              const Padding& padding) {
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        ConstEigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        EigenMatrixMap;

    ConstEigenMatrixMap in_mat(
        tensor_in.flat<T>().data(), params.depth,
        params.tensor_in_cols * params.tensor_in_rows * params.tensor_in_batch);
    ConstEigenMatrixMap out_mat(
        tensor_out.flat<T>().data(), params.depth,
        params.out_width * params.out_height * params.tensor_in_batch);
    ConstEigenMatrixMap top_diff_mat(
        top_diff.flat<T>().data(), params.depth,
        params.tensor_in_cols * params.tensor_in_rows * params.tensor_in_batch);
    EigenMatrixMap bottom_diff_mat(
        bottom_diff->flat<T>().data(), params.depth,
        params.out_width * params.out_height * params.tensor_in_batch);

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());

    // The following code basically does the following:
    // 1. Flattens the input, output, top_diff and bottom_diff tensors into
    //    two dimensional arrays.
    //    tensor_in_as_matrix:
    //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
    //    tensor_out_as_matrix:
    //      depth by (out_width * out_height * tensor_in_batch)
    //    top_diff_as_matrix:
    //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
    //    bottom_diff_as_matrix:
    //      depth by (out_width * out_height * tensor_in_batch)
    //
    // 2. Walks through the set of columns in the flattened
    //    tensor_in_as_matrix, tensor_out_as_matrix, top_diff_as_matrix
    //    and updates the column(s) corresponding to the maximum values in
    //    tensor_out_as_matrix with the corresponding values in
    //    top_diff_as_matrix.
    auto shard = [&params, &in_mat, &out_mat, &top_diff_mat, &bottom_diff_mat](
        int64 start, int64 limit) {
      const int32 depth = params.depth;
      const int32 in_rows = params.tensor_in_rows;
      const int32 in_cols = params.tensor_in_cols;
      const int32 pad_rows = params.pad_rows;
      const int32 pad_cols = params.pad_cols;
      const int32 window_rows = params.window_rows;
      const int32 window_cols = params.window_cols;
      const int32 row_stride = params.row_stride;
      const int32 col_stride = params.col_stride;
      const int32 out_height = params.out_height;
      const int32 out_width = params.out_width;

      {
        // Initializes the output grad backprop tensor with 0.
        const int32 output_image_size = out_height * out_width * params.depth;
        EigenMatrixMap bottom_diff_shard(
            bottom_diff_mat.data() + start * output_image_size, 1,
            (limit - start) * output_image_size);
        bottom_diff_shard.setZero();
      }

      for (int b = start; b < limit; ++b) {
        for (int ph = 0; ph < out_height; ++ph) {
          for (int pw = 0; pw < out_width; ++pw) {
            // (h_start, h_end) * (w_start, w_end) is the range that the input
            // vector projects to.
            int h_start = ph * row_stride - pad_rows;
            const int h_end = std::min(h_start + window_rows, in_rows);
            int w_start = pw * col_stride - pad_cols;
            const int w_end = std::min(w_start + window_cols, in_cols);
            h_start = std::max(h_start, 0);
            w_start = std::max(w_start, 0);
            const int out_index = (b * out_height + ph) * out_width + pw;
            // Find value corresponding to the input maximum in top_diff.
            for (int d = 0; d < depth; ++d) {
              const T& output_ref = out_mat.coeffRef(d, out_index);
              bool should_stop = false;
              for (int h = h_start; h < h_end && !should_stop; ++h) {
                for (int w = w_start; w < w_end && !should_stop; ++w) {
                  const int in_index = (b * in_rows + h) * in_cols + w;
                  const T& input_ref = in_mat.coeffRef(d, in_index);
                  if (output_ref == input_ref) {
                    T& bottom_diff_ref = bottom_diff_mat.coeffRef(d, out_index);
                    bottom_diff_ref = top_diff_mat.coeffRef(d, in_index);
                    should_stop = true;
                  }
                }
              }
            }
          }
        }
      }
    };

    const int64 shard_cost = params.out_width * params.out_height *
                             params.depth * params.window_rows *
                             params.window_cols;
    Shard(worker_threads.num_threads, worker_threads.workers,
          params.tensor_in_batch, shard_cost, shard);
  }

  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

#ifdef GOOGLE_CUDA

template <class T>
class MaxPoolingGradGradOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit MaxPoolingGradGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    if (context->num_inputs() == 3) {
      OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
      OP_REQUIRES(context, ksize_.size() == 4,
                  errors::InvalidArgument("Sliding window ksize field must "
                                          "specify 4 dimensions"));
      OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
      OP_REQUIRES(context, stride_.size() == 4,
                  errors::InvalidArgument("Sliding window strides field must "
                                          "specify 4 dimensions"));
      const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
      const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
      OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                  errors::Unimplemented(
                      "Pooling is not yet supported on the batch dimension."));
    }
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_grad_backprop = context->input(2);

    // For maxpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional 4"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    // For maxpooling, out_grad_backprop should have 4 dimensions.
    OP_REQUIRES(
        context, out_grad_backprop.dims() == 4,
        errors::InvalidArgument("out_grad_backprop must be 4-dimensional"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, tensor_out.shape(), &output));

    std::vector<int32> ksize = ksize_;
    std::vector<int32> stride = stride_;
    if (context->num_inputs() == 5) {
      const Tensor& tensor_ksize = context->input(3);
      auto value_ksize = tensor_ksize.flat<int32>();
      ksize.resize(tensor_ksize.shape().num_elements());
      std::copy_n(&value_ksize(0), ksize.size(), ksize.begin());

      const Tensor& tensor_stride = context->input(4);
      auto value_stride = tensor_stride.flat<int32>();
      stride.resize(tensor_stride.shape().num_elements());
      std::copy_n(&value_stride(0), stride.size(), stride.begin());
    }

    OP_REQUIRES(context, ksize.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, stride.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int32 ksize_n = GetTensorDim(ksize, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));

    PoolParameters params{context,  ksize,        stride,
                          padding_, data_format_, tensor_in.shape()};

    functor::MaxPoolGradBackwardNoMask<T>()(
        data_format_, tensor_in.flat<T>().data(), tensor_out.flat<T>().data(),
        params.tensor_in_batch, params.out_height, params.out_width,
        params.depth, params.tensor_in_rows, params.tensor_in_cols,
        params.window_rows, params.window_cols, params.row_stride,
        params.col_stride, params.pad_rows, params.pad_cols,
        out_grad_backprop.flat<T>().data(), output->flat<T>().data(),
        context->eigen_device<Eigen::GpuDevice>());
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool use_dnn_;
};

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
template <typename T>
struct LaunchMaxPoolingGradGradOpSYCL;

template <class T>
class MaxPoolingGradGradOp<SYCLDevice, T> : public OpKernel {
 public:
  explicit MaxPoolingGradGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    const int32 ksize_c = GetTensorDim(ksize_, data_format_, 'C');
    const int32 stride_c = GetTensorDim(stride_, data_format_, 'C');
    OP_REQUIRES(context, ksize_c == 1 && stride_c == 1,
                errors::Unimplemented("MaxPoolingGradGrad is not yet "
                                      "supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(1);
    const Tensor& out_grad_backprop = context->input(2);

    // For maxpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    // For maxpooling, out_grad_backprop should have 4 dimensions.
    OP_REQUIRES(
        context, out_grad_backprop.dims() == 4,
        errors::InvalidArgument("out_grad_backprop must be 4-dimensional"));

    PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, tensor_in.shape()};

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, tensor_out.shape(), &output));

    LaunchMaxPoolingGradGradOpSYCL<T>::launch(
        context, params, tensor_in, tensor_out, out_grad_backprop, output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};
#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename T>
struct LaunchMaxPoolingNoMask;

template <typename Device, typename T>
class MaxPoolingNoMaskOp : public OpKernel {
 public:
  explicit MaxPoolingNoMaskOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument(
            "Default MaxPoolingNoMaskOp only supports NHWC on device type ",
            DeviceTypeString(context->device_type())));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.out_height,
                           params.out_width, params.depth});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    LaunchMaxPoolingNoMask<Device, T>::launch(context, params, tensor_in,
                                              output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

template <typename Device, typename T>
class MaxPoolingNoMaskV2Op : public OpKernel {
 public:
  explicit MaxPoolingNoMaskV2Op(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument(
            "Default MaxPoolingNoMaskOp only supports NHWC on device type ",
            DeviceTypeString(context->device_type())));
    if (context->num_inputs() == 1) {
      OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
      OP_REQUIRES(context, ksize_.size() == 4,
                  errors::InvalidArgument("Sliding window ksize field must "
                                          "specify 4 dimensions"));
      OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
      OP_REQUIRES(context, stride_.size() == 4,
                  errors::InvalidArgument("Sliding window stride field must "
                                          "specify 4 dimensions"));
      OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                  errors::Unimplemented(
                      "Pooling is not yet supported on the batch dimension."));
    }
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    std::vector<int32> ksize = ksize_;
    std::vector<int32> stride = stride_;

    if (context->num_inputs() != 1) {
      const Tensor& tensor_ksize = context->input(1);
      auto value_ksize = tensor_ksize.flat<int32>();
      ksize.resize(tensor_ksize.shape().num_elements());
      std::copy_n(&value_ksize(0), ksize.size(), ksize.begin());

      const Tensor& tensor_stride = context->input(2);
      auto value_stride = tensor_stride.flat<int32>();
      stride.resize(tensor_stride.shape().num_elements());
      std::copy_n(&value_stride(0), stride.size(), stride.begin());
    }
    OP_REQUIRES(context, ksize.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, stride.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, ksize[0] == 1 && stride[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    PoolParameters params{context,  ksize,        stride,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.out_height,
                           params.out_width, params.depth});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    LaunchMaxPoolingNoMask<Device, T>::launch(context, params, tensor_in,
                                              output);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

template <typename Device, typename T>
struct LaunchMaxPoolingWithArgmax;

template <typename Device, typename T>
class MaxPoolingWithArgmaxOp : public OpKernel {
 public:
  explicit MaxPoolingWithArgmaxOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    PoolParameters params{context,  ksize_,      stride_,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.out_height,
                           params.out_width, params.depth});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    Tensor* argmax = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, out_shape, &argmax));

    LaunchMaxPoolingWithArgmax<Device, T>::launch(context, params, tensor_in,
                                                  output, argmax);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

template <typename Device, typename T>
struct LaunchMaxPoolingGradWithArgmax;

template <typename Device, typename T>
class MaxPoolingGradWithArgmaxOp : public OpKernel {
 public:
  explicit MaxPoolingGradWithArgmaxOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format_str;
    auto status = context->GetAttr("data_format", &data_format_str);
    if (status.ok()) {
      OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    }

    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& grad_in = context->input(1);
    const Tensor& argmax = context->input(2);

    PoolParameters params{context,  ksize_,      stride_,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.tensor_in_rows,
                           params.tensor_in_cols, params.depth});
    Tensor* grad_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {1}, 0, out_shape, &grad_out));

    LaunchMaxPoolingGradWithArgmax<Device, T>::launch(context, params, grad_in,
                                                      argmax, grad_out);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

template <typename Device, typename T>
struct LaunchMaxPoolingGradGradWithArgmax;

template <typename Device, typename T>
class MaxPoolingGradGradWithArgmaxOp : public OpKernel {
 public:
  explicit MaxPoolingGradGradWithArgmaxOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& grad_in = context->input(1);
    const Tensor& argmax = context->input(2);

    PoolParameters params{context,  ksize_,      stride_,
                          padding_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape({params.tensor_in_batch, params.out_height,
                           params.out_width, params.depth});

    Tensor* grad_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {1}, 0, out_shape, &grad_out));

    LaunchMaxPoolingGradGradWithArgmax<Device, T>::launch(
        context, params, grad_in, argmax, grad_out);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
};

#if GOOGLE_CUDA
template <typename T>
class MaxPoolingNoMaskOp<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit MaxPoolingNoMaskOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
    use_dnn_ = CanUseCudnn();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape =
        ShapeFromFormat(data_format_, params.tensor_in_batch, params.out_height,
                        params.out_width, params.depth);

    // Assuming qint8 <--> NCHW_VECT_C (int8x4) here.
    constexpr bool is_int8x4 = std::is_same<T, qint8>::value;
    OP_REQUIRES(context, (is_int8x4 == (data_format_ == FORMAT_NCHW_VECT_C)),
                errors::InvalidArgument(
                    "qint8 should be used with data_format NCHW_VECT_C."));

    // These is_int8x4 checks avoid linker errors for missing qint8 kernels.
    if (!is_int8x4 && use_dnn_ && data_format_ == FORMAT_NCHW) {
      DnnPoolingOp<T>::Compute(
          context, perftools::gputools::dnn::PoolingMode::kMaximum, ksize_,
          stride_, padding_, data_format_, tensor_in, out_shape);
    } else {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
      if (is_int8x4) {
        LaunchMaxPoolingNoMask_NCHW_VECT_C<Device>::launch(context, params,
                                                           tensor_in, output);
      } else if (data_format_ == FORMAT_NHWC) {
        LaunchMaxPoolingNoMask<Device, T>::launch(context, params, tensor_in,
                                                  output);
      } else {
        LOG(FATAL) << "MaxPool currently only supports the following (layout, "
                      "type) combinations: (NHWC, non-qint8), "
                      "(NCHW, non-qint8) or (NCHW_VECT_C, qint8). The "
                      "requested combination ("
                   << ToString(data_format_) << ", "
                   << DataTypeString(DataTypeToEnum<T>::v())
                   << ") is not supported.";
      }
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool use_dnn_;
};

template <typename T>
class MaxPoolingNoMaskV2Op<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit MaxPoolingNoMaskV2Op(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    if (context->num_inputs() == 1) {
      OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
      OP_REQUIRES(context, ksize_.size() == 4,
                  errors::InvalidArgument("Sliding window ksize field must "
                                          "specify 4 dimensions"));
      OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
      OP_REQUIRES(context, stride_.size() == 4,
                  errors::InvalidArgument("Sliding window stride field must "
                                          "specify 4 dimensions"));
      const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
      const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
      OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                  errors::Unimplemented(
                      "Pooling is not yet supported on the batch dimension."));
    }
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    use_dnn_ = CanUseCudnn();
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    std::vector<int32> ksize = ksize_;
    std::vector<int32> stride = stride_;

    if (context->num_inputs() != 1) {
      const Tensor& tensor_ksize = context->input(1);
      auto value_ksize = tensor_ksize.flat<int32>();
      ksize.resize(tensor_ksize.shape().num_elements());
      std::copy_n(&value_ksize(0), ksize.size(), ksize.begin());

      const Tensor& tensor_stride = context->input(2);
      auto value_stride = tensor_stride.flat<int32>();
      stride.resize(tensor_stride.shape().num_elements());
      std::copy_n(&value_stride(0), stride.size(), stride.begin());
    }
    OP_REQUIRES(context, ksize.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, stride.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    const int32 ksize_n = GetTensorDim(ksize, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));

    PoolParameters params{context,  ksize,        stride,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    TensorShape out_shape =
        ShapeFromFormat(data_format_, params.tensor_in_batch, params.out_height,
                        params.out_width, params.depth);
    if (use_dnn_ && data_format_ == FORMAT_NCHW) {
      DnnPoolingOp<T>::Compute(
          context, perftools::gputools::dnn::PoolingMode::kMaximum, ksize,
          stride, padding_, data_format_, tensor_in, out_shape);
    } else {
      CHECK(data_format_ == FORMAT_NHWC)
          << "Non-Cudnn MaxPool only supports NHWC format";
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
      LaunchMaxPoolingNoMask<Device, T>::launch(context, params, tensor_in,
                                                output);
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool use_dnn_;
};

template <typename T>
struct LaunchMaxPoolingNoMask<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& input, Tensor* output) {
    bool status = functor::MaxPoolForwardWithOptionalArgmax<T>()(
        input.flat<T>().data(), params.tensor_in_batch, params.tensor_in_rows,
        params.tensor_in_cols, params.depth, params.out_height,
        params.out_width, params.window_rows, params.window_cols,
        params.row_stride, params.col_stride, params.pad_rows, params.pad_cols,
        output->flat<T>().data(), nullptr, context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPoolForwardNoMask"));
    }
  }
};

template <typename T>
struct LaunchMaxPoolingWithArgmax<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& input, Tensor* output, Tensor* argmax) {
    bool status = functor::MaxPoolForwardWithOptionalArgmax<T>()(
        input.flat<T>().data(), params.tensor_in_batch, params.tensor_in_rows,
        params.tensor_in_cols, params.depth, params.out_height,
        params.out_width, params.window_rows, params.window_cols,
        params.row_stride, params.col_stride, params.pad_rows, params.pad_cols,
        output->flat<T>().data(),
        reinterpret_cast<int64*>(argmax->flat<int64>().data()),
        context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPoolForwardWithArgmax"));
    }
  }
};

template <typename T>
struct LaunchMaxPoolingGradWithArgmax<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& grad_in, const Tensor& argmax,
                     Tensor* grad_out) {
    const int input_size = params.tensor_in_batch * params.tensor_in_rows *
                           params.tensor_in_cols * params.depth;
    const int output_size = params.tensor_in_batch * params.out_height *
                            params.out_width * params.depth;
    const int top_offset = params.out_height * params.out_width * params.depth;
    const int bottom_offset =
        params.tensor_in_rows * params.tensor_in_cols * params.depth;
    bool status = functor::MaxPoolBackwardWithArgmax<T>()(
        output_size, input_size, grad_in.flat<T>().data(),
        reinterpret_cast<const int64*>(argmax.flat<int64>().data()), top_offset,
        bottom_offset, grad_out->flat<T>().data(), context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPoolBackwardWithArgmax"));
    }
  }
};

template <typename T>
struct LaunchMaxPoolingGradGradWithArgmax<Eigen::GpuDevice, T> {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& grad_in, const Tensor& argmax,
                     Tensor* grad_out) {
    const int input_size = params.tensor_in_batch * params.tensor_in_rows *
                           params.tensor_in_cols * params.depth;
    const int output_size = params.tensor_in_batch * params.out_height *
                            params.out_width * params.depth;
    const int top_offset =
        params.tensor_in_rows * params.tensor_in_cols * params.depth;
    const int bottom_offset =
        params.out_width * params.out_height * params.depth;
    bool status = functor::MaxPoolGradBackwardWithArgmax<T>()(
        output_size, input_size, grad_in.flat<T>().data(),
        reinterpret_cast<const int64*>(argmax.flat<int64>().data()), top_offset,
        bottom_offset, grad_out->flat<T>().data(), context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching MaxPoolGradBackwardWithArgmax"));
    }
  }
};

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
// MaxPoolGrad SYCL kernel. Expects the number of threads to be equal to the
// number of elements in the output backprop tenor (i.e. the number of elements
// in the input data tensor).
//
// For each output backprop element we compute the possible window of values in
// the input backprop tensor which might contribute to this element. Then for
// each error in this window, compute the corresponding input window which was
// pooled into that element in the output. Walk through this input window to
// determine whether the input value is the first maximum value, and so the
// error should be propagated back to the corresponding backprop element.
template <typename T>
class MaxPoolGradSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  MaxPoolGradSYCL(const int depth, const int batch, const int in_rows,
                  const int in_cols, const std::array<int64, 2>& output_shape,
                  const std::array<int64, 2>& window,
                  const std::array<int64, 2>& stride,
                  const std::array<int64, 2>& padding,
                  const read_accessor input_data_accessor,
                  const read_accessor output_data_accessor,
                  const read_accessor input_backprop_accessor,
                  write_accessor output_backprop_accessor)
      : p_(depth, batch, in_rows, in_cols, output_shape, window, stride,
           padding),
        input_data_accessor_(input_data_accessor),
        output_data_accessor_(output_data_accessor),
        input_backprop_accessor_(input_backprop_accessor),
        output_backprop_accessor_(output_backprop_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_data_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_data_accessor_);
    T* input_backprop = ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);

    const int index = item.get_linear_id();
    T output_value = 0;
    int n = index;
    const int d = n % p_.depth_;
    n /= p_.depth_;
    const int c = (n % p_.in_cols_) + p_.pad_cols_;
    const int poolcstart =
        (c < p_.window_cols_) ? 0 : (c - p_.window_cols_) / p_.stride_cols_ + 1;
    const int poolcend = std::min(c / p_.stride_cols_ + 1, p_.out_cols_);
    n /= p_.in_cols_;
    const int r = (n % p_.in_rows_) + p_.pad_rows_;
    const int poolrstart =
        (r < p_.window_rows_) ? 0 : (r - p_.window_rows_) / p_.stride_rows_ + 1;
    const int poolrend = std::min(r / p_.stride_rows_ + 1, p_.out_rows_);
    n /= p_.in_rows_;
    const int index_no_n = index - n * p_.in_cols_ * p_.in_rows_ * p_.depth_;

    const T* input_data_n =
        input_data + n * p_.in_cols_ * p_.in_rows_ * p_.depth_;
    const T* output_data_n =
        output_data + n * p_.out_cols_ * p_.out_rows_ * p_.depth_;
    const T* input_backprop_n =
        input_backprop + n * p_.out_cols_ * p_.out_rows_ * p_.depth_;

    for (int poolr = poolrstart; poolr < poolrend; ++poolr) {
      int rstart = poolr * p_.stride_rows_ - p_.pad_rows_;
      const int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
      rstart = std::max(rstart, 0);

      for (int poolc = poolcstart; poolc < poolcend; ++poolc) {
        int cstart = poolc * p_.stride_cols_ - p_.pad_cols_;
        const int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
        cstart = std::max(cstart, 0);

        const int output_data_idx =
            (poolr * p_.out_cols_ + poolc) * p_.depth_ + d;
        bool should_continue = true;
        bool is_max = (input_data[index] == output_data_n[output_data_idx]);
        for (int win_r = rstart; win_r < rend && should_continue; ++win_r) {
          for (int win_c = cstart; win_c < cend && should_continue; ++win_c) {
            const int input_data_idx =
                (win_r * p_.in_cols_ + win_c) * p_.depth_ + d;
            if (input_data_idx == index_no_n) {
              should_continue = false;
            } else if (input_data_n[input_data_idx] ==
                       output_data_n[output_data_idx]) {
              should_continue = false;
              is_max = false;
            }
          }
        }
        if (is_max) {
          output_value += input_backprop_n[output_data_idx];
        }
      }
    }
    output_backprop[index] = output_value;
  }

 private:
  const SYCL2DPoolParams p_;

  const read_accessor input_data_accessor_;
  const read_accessor output_data_accessor_;
  const read_accessor input_backprop_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T>
struct LaunchMaxPoolingGradOpSYCL {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const Tensor& tensor_out, const Tensor& out_backprop,
                     const std::array<int64, 2>& window,
                     const std::array<int64, 2>& stride,
                     const std::array<int64, 2>& out,
                     const std::array<int64, 2>& padding,
                     TensorFormat data_format, Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    const int batch = GetTensorDim(tensor_in, data_format, 'N');
    const int in_rows = GetTensorDim(tensor_in, data_format, '0');
    const int in_cols = GetTensorDim(tensor_in, data_format, '1');
    const int depth = GetTensorDim(tensor_in, data_format, 'C');

    const int output_size = output->NumElements();

    auto input_data_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_data_buffer =
        device.get_sycl_buffer(tensor_out.template flat<T>().data());
    auto input_backprop_buffer =
        device.get_sycl_buffer(out_backprop.template flat<T>().data());
    auto output_backprop_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_data_access =
          input_data_buffer.template get_access<cl::sycl::access::mode::read>(
              cgh);
      auto output_data_access =
          output_data_buffer.template get_access<cl::sycl::access::mode::read>(
              cgh);
      auto input_backprop_access =
          input_backprop_buffer
              .template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_backprop_access =
          output_backprop_buffer
              .template get_access<cl::sycl::access::mode::write>(cgh);
      MaxPoolGradSYCL<T> max_pool(depth, batch, in_rows, in_cols, out, window,
                                  stride, padding, input_data_access,
                                  output_data_access, input_backprop_access,
                                  output_backprop_access);

      cgh.parallel_for(cl::sycl::range<1>(output_size), max_pool);
    });
  }
};
// MaxPoolGradGrad SYCL kernel. Expects the number of threads to be equal to
// the number of elements in the output backprop tensor, i.e. the number of
// elements in the output tensor.
//
// For each element in the output backprop tensor, find the corresponding input
// window, and compare the input and output data to find the index of the
// maximum value in the input tensor. This is then the index of the gradient to
// pass through to the output backprop tensor.
template <typename T>
class MaxPoolGradGradSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

 public:
  MaxPoolGradGradSYCL(const PoolParameters& params,
                      const read_accessor input_data_accessor,
                      const read_accessor output_data_accessor,
                      const read_accessor input_backprop_accessor,
                      write_accessor output_backprop_accessor)
      : p_(params),
        input_data_accessor_(input_data_accessor),
        output_data_accessor_(output_data_accessor),
        input_backprop_accessor_(input_backprop_accessor),
        output_backprop_accessor_(output_backprop_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_data_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_data_accessor_);
    T* input_backprop = ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);

    int index = item.get_linear_id();
    int n = index;
    int d = n % p_.depth_;
    n /= p_.depth_;
    int cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
    int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
    cstart = std::max(cstart, 0);
    n /= p_.out_cols_;
    int rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
    int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
    rstart = std::max(rstart, 0);
    n /= p_.out_rows_;
    int maxidx = -1;
    bool should_stop = false;
    const T* input_data_n =
        input_data + n * p_.in_cols_ * p_.in_rows_ * p_.depth_;
    for (int r = rstart; r < rend && !should_stop; ++r) {
      for (int c = cstart; c < cend && !should_stop; ++c) {
        int idx = (r * p_.in_cols_ + c) * p_.depth_ + d;
        if (output_data[index] == input_data_n[idx]) {
          maxidx = idx;
          should_stop = true;
        }
      }
    }
    if (maxidx != -1) {
      output_backprop[index] =
          input_backprop[n * p_.in_rows_ * p_.in_cols_ * p_.depth_ + maxidx];
    }
  }

 private:
  const SYCL2DPoolParams p_;

  const read_accessor input_data_accessor_;
  const read_accessor output_data_accessor_;
  const read_accessor input_backprop_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T>
struct LaunchMaxPoolingGradGradOpSYCL {
  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& tensor_in, const Tensor& tensor_out,
                     const Tensor& out_backprop, Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();

    const int num_threads = output->NumElements();

    auto input_data_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_data_buffer =
        device.get_sycl_buffer(tensor_out.template flat<T>().data());
    auto input_backprop_buffer =
        device.get_sycl_buffer(out_backprop.template flat<T>().data());
    auto output_backprop_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_data_access =
          input_data_buffer.template get_access<cl::sycl::access::mode::read>(
              cgh);
      auto output_data_access =
          output_data_buffer.template get_access<cl::sycl::access::mode::read>(
              cgh);
      auto input_backprop_access =
          input_backprop_buffer
              .template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_backprop_access =
          output_backprop_buffer
              .template get_access<cl::sycl::access::mode::write>(cgh);
      MaxPoolGradGradSYCL<T> maxpoolgradgrad(
          params, input_data_access, output_data_access, input_backprop_access,
          output_backprop_access);

      cgh.parallel_for(cl::sycl::range<1>(num_threads), maxpoolgradgrad);
    });
  }
};
#endif  // TENSORFLOW_USE_SYCL

#define REGISTER_MAX_POOL_KERNELS(D, T)                                  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("MaxPoolGrad").Device(DEVICE_##D).TypeConstraint<T>("T"),     \
      MaxPoolingGradOp<D##Device, T>);                                   \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("MaxPoolGradGrad").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      MaxPoolingGradGradOp<D##Device, T>);                               \
  REGISTER_KERNEL_BUILDER(Name("MaxPoolGradV2")                          \
                              .Device(DEVICE_##D)                        \
                              .HostMemory("ksize")                       \
                              .HostMemory("strides")                     \
                              .TypeConstraint<T>("T"),                   \
                          MaxPoolingGradOp<D##Device, T>);               \
  REGISTER_KERNEL_BUILDER(Name("MaxPoolGradGradV2")                      \
                              .Device(DEVICE_##D)                        \
                              .HostMemory("ksize")                       \
                              .HostMemory("strides")                     \
                              .TypeConstraint<T>("T"),                   \
                          MaxPoolingGradGradOp<D##Device, T>);

// Below kernels implemented only for CPU device.
#define REGISTER_CPU_ONLY_POOL_KERNELS(T)                          \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("MaxPool").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      MaxPoolingOp<CPUDevice, T>);                                 \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("MaxPoolV2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MaxPoolingV2Op<CPUDevice, T>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_ONLY_POOL_KERNELS);
#undef REGISTER_CPU_ONLY_POOL_KERNELS

#define REGISTER_CPU_MAX_POOL_KERNELS(T) REGISTER_MAX_POOL_KERNELS(CPU, T);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_MAX_POOL_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA

// Forward declarations for the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                            \
  template <>                                                          \
  void SpatialMaxPooling<Eigen::GpuDevice, T>::operator()(             \
      const Eigen::GpuDevice& d, typename TTypes<T, 4>::Tensor output, \
      typename TTypes<T, 4>::ConstTensor input, int window_rows,       \
      int window_cols, int row_stride, int col_stride,                 \
      const Eigen::PaddingType& padding);                              \
  extern template struct SpatialMaxPooling<Eigen::GpuDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_GPU_MAX_POOL_KERNELS(T) REGISTER_MAX_POOL_KERNELS(GPU, T)
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_MAX_POOL_KERNELS);
#undef REGISTER_GPU_MAX_POOL_KERNELS

// Below kernels currently implemented only for GPU device.
// Note(jiayq): Currently, the Caffe custom implementation is faster than the
// default Eigen implementation so we are using the custom kernel as the
// default. However, you can explicitly invoke the eigen version using
// kernel_label_map.
#define REGISTER_GPU_ONLY_POOL_KERNELS(T)                            \
  REGISTER_KERNEL_BUILDER(Name("MaxPool")                            \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<T>("T")                \
                              .Label("eigen_tensor"),                \
                          MaxPoolingOp<GPUDevice, T>);               \
  REGISTER_KERNEL_BUILDER(Name("MaxPoolV2")                          \
                              .Device(DEVICE_GPU)                    \
                              .HostMemory("ksize")                   \
                              .HostMemory("strides")                 \
                              .TypeConstraint<T>("T")                \
                              .Label("eigen_tensor"),                \
                          MaxPoolingV2Op<GPUDevice, T>);             \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("MaxPool").Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
      MaxPoolingNoMaskOp<GPUDevice, T>);                             \
  REGISTER_KERNEL_BUILDER(Name("MaxPoolV2")                          \
                              .Device(DEVICE_GPU)                    \
                              .HostMemory("ksize")                   \
                              .HostMemory("strides")                 \
                              .TypeConstraint<T>("T"),               \
                          MaxPoolingNoMaskV2Op<GPUDevice, T>);       \
  REGISTER_KERNEL_BUILDER(Name("MaxPoolWithArgmax")                  \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<int64>("Targmax")      \
                              .TypeConstraint<T>("T"),               \
                          MaxPoolingWithArgmaxOp<GPUDevice, T>);     \
  REGISTER_KERNEL_BUILDER(Name("MaxPoolGradWithArgmax")              \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<int64>("Targmax"),     \
                          MaxPoolingGradWithArgmaxOp<GPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("MaxPoolGradGradWithArgmax")          \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<int64>("Targmax"),     \
                          MaxPoolingGradGradWithArgmaxOp<GPUDevice, T>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_ONLY_POOL_KERNELS);

// TODO(b/65847473): Re-enable once the underlying build error is fixed.
#if !defined(PLATFORM_WINDOWS)
REGISTER_KERNEL_BUILDER(
    Name("MaxPool").Device(DEVICE_GPU).TypeConstraint<qint8>("T"),
    MaxPoolingNoMaskOp<GPUDevice, qint8>);

REGISTER_KERNEL_BUILDER(Name("MaxPoolV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("ksize")
                            .HostMemory("strides")
                            .TypeConstraint<qint8>("T"),
                        MaxPoolingV2Op<GPUDevice, qint8>);

REGISTER_KERNEL_BUILDER(Name("MaxPoolV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("ksize")
                            .HostMemory("strides")
                            .TypeConstraint<qint8>("T")
                            .Label("eigen_tensor"),
                        MaxPoolingV2Op<GPUDevice, qint8>);
#endif  // !defined(PLATFORM_WINDOWS)

#undef REGISTER_GPU_ONLY_POOL_KERNELS

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("MaxPool")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<float>("T"),
                        MaxPoolingOp<SYCLDevice, float>);
#define REGISTER_SYCL_MAX_POOL_KERNELS(T) REGISTER_MAX_POOL_KERNELS(SYCL, T)
TF_CALL_float(REGISTER_SYCL_MAX_POOL_KERNELS);
#undef REGISTER_SYCL_MAX_POOL_KERNELS
#endif  // TENSORFLOW_USE_SYCL
#undef REGISTER_MAX_POOL_KERNELS

}  // namespace tensorflow
