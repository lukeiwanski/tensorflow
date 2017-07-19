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

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_3d.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include <iostream>

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
using perftools::gputools::dnn::DimIndex;
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif

template <typename Device, typename T>
struct LaunchConvOp;

template <typename T>
struct LaunchConvOp<CPUDevice, T> {
  static void launch(OpKernelContext* context, bool cudnn_use_autotune,
                     const Tensor& input, const Tensor& filter,
                     const std::array<int64, 3>& strides, const Padding padding,
                     TensorFormat data_format, Tensor* output) {
    OP_REQUIRES(context, data_format == FORMAT_NHWC,
                errors::InvalidArgument("CPU implementation of Conv3D "
                                        "currently only supports the NHWC "
                                        "tensor format."));
    std::cout << "CPU input: " << input.SummarizeValue(100) << std::endl;
    std::cout << "xCPU filter: " << filter.SummarizeValue(100) << std::endl;
    functor::CuboidConvolution<CPUDevice, T>()(
        context->eigen_device<CPUDevice>(), output->tensor<T, 5>(),
        input.tensor<T, 5>(), filter.tensor<T, 5>(), strides[2], strides[1],
        strides[0], BrainPadding2EigenPadding(padding));
    std::cout << "CPU output: " << output->SummarizeValue(100) << std::endl;
  }
};

#ifdef TENSORFLOW_USE_SYCL
// template <typename T>
// class Conv3DSYCL {
//   using write_accessor =
//       cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
//                          cl::sycl::access::target::global_buffer>;
//   using read_accessor =
//       cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
//                          cl::sycl::access::target::global_buffer>;
//
//  public:
//   Conv3DSYCL(const int64 batch, const int64 in_planes, const int64 in_rows,
//                 const int64 in_cols,  const int64 in_depth,
//                 const int64 filter_planes, const int64 filter_rows,
//                 const int64 filter_cols, const int64 out_depth,
//                 const int64 pad_planes, const int64 pad_rows,
//                 const int64 pad_cols, const int64 out_planes,
//                 const int64 out_rows, const int64 out_cols,
//                 read_accessor input_accessor, read_accessor filter_accessor,
//                 write_accessor output_accessor)
//       : batch_(batch),
//         in_planes_(in_planes),
//         in_rows_(in_rows),
//         in_cols_(in_cols),
//         in_depth_(in_depth),
//         filter_planes_(filter_planes),
//         filter_rows_(filter_rows),
//         filter_cols_(filter_cols),
//         out_depth_(out_depth),
//         pad_planes_(pad_planes),
//         pad_rows_(pad_rows),
//         pad_cols_(pad_cols),
//         out_planes_(out_planes),
//         out_rows_(out_rows),
//         out_cols_(out_cols),
//         input_accessor_(input_accessor),
//         filter_accessor_(filter_accessor),
//         output_accessor_(output_accessor) {}
//   void operator()(cl::sycl::item<1> item) {
//     //stride = 1
//     T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
//     T* filter_data = ConvertToActualTypeSycl(T, filter_accessor_);
//     T* output_data = ConvertToActualTypeSycl(T, output_accessor_);
//
//     int index = item.get_linear_id();
//     int n = index;
//     int d = n % out_depth_;
//     n /= out_depth_;
//     int cstart = (n % out_cols_) - pad_cols_;
//     int cend = std::min(cstart + filter_cols_, in_cols_);
//     cstart = std::max(cstart, 0);
//     n /= out_cols_;
//     int rstart = (n % out_rows_) - pad_rows_;
//     int rend = std::min(rstart + filter_rows_, in_rows_);
//     rstart = std::max(rstart, 0);
//     n /= out_rows_;
//     int pstart = (n % out_planes_) - pad_planes_;
//     int pend = std::min(pstart + filter_planes_, in_planes_);
//     pstart = std::max(pstart, 0);
//     n /= out_planes_;
//
//     T sum = T(0);
//     const T* input_data_n =
//         input_data + n * in_planes_ * in_cols_ * in_rows_ * in_depth_;
//     const T* filter_data_n =
//         filter_data + n * filter_planes_ * filter_cols_ * filter_rows_ * out_depth_;
//     for (int p = pstart; p < pend; ++p) {
//       for (int r = rstart; r < rend; ++r) {
//         for (int c = cstart; c < cend; ++c) {
//           int idx = ((p * in_rows_ + r) * in_cols_ + c) * in_depth_ + d;
//           int filter_offset
//               = ((p * filter_rows_ + r) * filter_cols_ + c) * out_depth_ + d;
//           sum += input_data_n[idx] * filter_data_n[filter_offset];
//         }
//       }
//     }
//     T* output_data_n =
//         output_data + n * out_planes_ * out_cols_ * out_rows_ * out_depth_;
//     int pval = (pstart+pend-1)/2;
//     int rval = (rstart+rend-1)/2;
//     int cval = (cstart+cend-1)/2;
//     int out_idx = ((pval * out_rows_ + rval) * out_cols_ + cval) * out_depth_ + d;
//     output_data_n[out_idx] = sum;
//   }
//
//  private:
//   const int64 batch_;
//   const int64 in_planes_;
//   const int64 in_rows_;
//   const int64 in_cols_;
//   const int64 in_depth_;
//   const int64 filter_planes_;
//   const int64 filter_rows_;
//   const int64 filter_cols_;
//   const int64 out_depth_;
//   const int64 pad_planes_;
//   const int64 pad_rows_;
//   const int64 pad_cols_;
//   const int64 out_planes_;
//   const int64 out_rows_;
//   const int64 out_cols_;
//   const read_accessor input_accessor_;
//   const read_accessor filter_accessor_;
//   write_accessor output_accessor_;
// };

template <typename T>
struct LaunchConvOp<SYCLDevice, T> {
  static void launch(OpKernelContext* context, bool cudnn_use_autotune,
                     const Tensor& input, const Tensor& filter,
                     const std::array<int64, 3>& strides, const Padding padding,
                     TensorFormat data_format, Tensor* output) {
    OP_REQUIRES(context, data_format == FORMAT_NHWC,
                errors::InvalidArgument("SYCL implementation of Conv3D "
                                        "currently only supports the NHWC "
                                        "tensor format."));
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    Tensor input_tensor = input;
    Tensor filter_tensor = filter;

    const int64 batch_ = GetTensorDim(input_tensor, data_format, 'N');
    int64 in_planes_ = GetTensorDim(input_tensor, data_format, '0');
    int64 in_rows_ = GetTensorDim(input_tensor, data_format, '1');
    int64 in_cols_ = GetTensorDim(input_tensor, data_format, '2');
    const int64 in_depth_ = GetTensorDim(input_tensor, data_format, 'C');

    // int64 filter_planes_ = GetTensorDim(filter_tensor, data_format, '0');
    // int64 filter_rows_ = GetTensorDim(filter_tensor, data_format, '1');
    // int64 filter_cols_ = GetTensorDim(filter_tensor, data_format, '2');
    const int64 filter_depth_ = GetTensorDim(filter_tensor, data_format, 'C');

    const int64 filter_planes_ = filter.dim_size(0);
    const int64 filter_rows_ = filter.dim_size(1);
    const int64 filter_cols_ = filter.dim_size(2);
    const int64 out_depth_ = filter.dim_size(4);

    int64 pad_planes_ = 0, pad_rows_ = 0, pad_cols_ = 0;
    int64 out_planes_ = GetTensorDim(*output, data_format, '0');
    int64 out_rows_ = GetTensorDim(*output, data_format, '1');
    int64 out_cols_ = GetTensorDim(*output, data_format, '2');

    if (padding == Padding::SAME) {
      pad_planes_ = std::max<int64>(
          0, (out_planes_ - 1) * strides[0] + filter.dim_size(0) - in_planes_);
      pad_rows_ = std::max<int64>(
          0, (out_rows_ - 1) * strides[1] + filter.dim_size(1) - in_rows_);
      pad_cols_ = std::max<int64>(
          0, (out_cols_ - 1) * strides[2] + filter.dim_size(2) - in_cols_);
    }

    // std::cout << "batch: " << batch << std::endl
    //           << "in_planes: " << in_planes << std::endl
    //           << "in_rows: " << in_rows << std::endl
    //           << "in_cols: " << in_cols << std::endl
    //           << "in_depth: " << in_depth << std::endl
    //           << "filter_planes: " << filter_planes << std::endl
    //           << "filter_rows: " << filter_rows << std::endl
    //           << "filter_cols: " << filter_cols << std::endl
    //           << "out_depth: " << out_depth << std::endl
    //           << "pad_planes: " << pad_planes << std::endl
    //           << "pad_rows: " << pad_rows << std::endl
    //           << "pad_cols: " << pad_cols << std::endl
    //           << "out_planes: " << out_planes << std::endl
    //           << "out_rows: " << out_rows << std::endl
    //           << "out_cols: " << out_cols << std::endl;

    std::cout << "batch: " << batch_ << std::endl
              << "in_planes: " << in_planes_ << std::endl
              << "in_rows: " << in_rows_ << std::endl
              << "in_cols: " << in_cols_ << std::endl
              << "in_depth: " << in_depth_ << std::endl
              << "filter_planes: " << filter_planes_ << std::endl
              << "filter_rows: " << filter_rows_ << std::endl
              << "filter_cols: " << filter_cols_ << std::endl
              << "filter_depth: " << filter_depth_ << std::endl
              << "out_depth: " << out_depth_ << std::endl
              << "pad_planes: " << pad_planes_ << std::endl
              << "pad_rows: " << pad_rows_ << std::endl
              << "pad_cols: " << pad_cols_ << std::endl
              << "out_planes: " << out_planes_ << std::endl
              << "out_rows: " << out_rows_ << std::endl
              << "out_cols: " << out_cols_ << std::endl;

    int num_threads = output->NumElements();
    std::cout << "num_threads: " << num_threads << std::endl;

    // auto input_buffer =
    //     device.get_sycl_buffer(input.template flat<T>().data());
    // auto filter_buffer =
    //     device.get_sycl_buffer(filter.template flat<T>().data());
    // auto output_buffer =
    //     device.get_sycl_buffer(output->template flat<T>().data());
    //
    // device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
    //   auto input_access =
    //       input_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
    //   auto filter_access =
    //       filter_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
    //   auto output_access =
    //       output_buffer.template get_access<cl::sycl::access::mode::write>(cgh);
    //   Conv3DSYCL<T> functor(batch, in_planes, in_rows, in_cols,  in_depth,
    //                         filter_planes, filter_rows, filter_cols, out_depth,
    //                         pad_planes, pad_rows, pad_cols, out_planes, out_rows,
    //                         out_cols,input_access, filter_access, output_access);
    //
    //   cgh.parallel_for(cl::sycl::range<1>(num_threads), functor);
    // });

    auto input_data = input.template flat<T>().data();
    auto filter_data = filter.template flat<T>().data();
    auto output_data = output->template flat<T>().data();
    for(int index = 0; index < num_threads; ++index){
      int n = index;
      int d = n % out_depth_;
      n /= out_depth_;
      int cstart = (n % out_cols_) * strides[0] - pad_cols_;
      int cend = std::min(cstart + filter_cols_, in_cols_);
      cstart = std::max(cstart, 0);
      n /= out_cols_;
      int rstart = (n % out_rows_) * strides[1] - pad_rows_;
      int rend = std::min(rstart + filter_rows_, in_rows_);
      rstart = std::max(rstart, 0);
      n /= out_rows_;
      int pstart = (n % out_planes_) * strides[2] - pad_planes_;
      int pend = std::min(pstart + filter_planes_, in_planes_);
      pstart = std::max(pstart, 0);
      n /= out_planes_;
      std::cout << cstart << "-" << cend << ", "
                << rstart << "-" << rend << ", "
                << pstart << "-" << pend << ", "
                << d << std::endl;
      const T* input_data_n =
          input_data + n * in_planes_ * in_cols_ * in_rows_ * in_depth_;
      const T* filter_data_n =
          filter_data + n * filter_planes_ * filter_cols_ * filter_rows_ * out_depth_;
      int pval = (pstart+pend-1)/2;
      int rval = (rstart+rend-1)/2;
      int cval = (cstart+cend-1)/2;
      for (int ptemp = pstart; ptemp < pend; ++ptemp) {
        T sum = T(0);
        std::cout << "sum: ";
        for (int rtemp = rstart; rtemp < rend; ++rtemp) {
          for (int ctemp = cstart; ctemp < cend; ++ctemp) {
            for(int dtemp = 0; dtemp < in_depth_; ++dtemp){
              int idx = ((ptemp * in_rows_ + rtemp) * in_cols_ + ctemp) * in_depth_ + dtemp;
              int p_off = ptemp % filter_planes_;
              int c_off = ctemp % filter_cols_;
              int r_off = rtemp % filter_rows_;
              int d_off = dtemp % filter_depth_;
              int filter_offset
                  = ((p_off * filter_rows_ + d_off) * filter_cols_ + c_off) * filter_depth_ + d;
              sum += input_data_n[idx] * filter_data[filter_offset];
              std::cout << input_data_n[idx] << "*" << filter_data[filter_offset]
                        << "+";
            }
          }
        }
        std::cout << "=" << sum << std::endl;
        T* output_data_n =
            output_data + n * out_planes_ * out_cols_ * out_rows_ * out_depth_;
        int out_idx = ((pval * out_rows_ + rval) * out_cols_ + cval) * out_depth_ + d;
        output_data_n[out_idx] = sum;
      }
    }

    // std::cout << "SYCL input: " << input.SummarizeValue(10) << std::endl;
    // std::cout << "SYCL filter: " << filter.SummarizeValue(10) << std::endl;
    // functor::CuboidConvolution<SYCLDevice, T>()(
    //     context->eigen_device<SYCLDevice>(), output->tensor<T, 5>(),
    //     input.tensor<T, 5>(), filter.tensor<T, 5>(), strides[2], strides[1],
    //     strides[0], BrainPadding2EigenPadding(padding));
    // std::cout << "SYCL output: " << output->SummarizeValue(10) << std::endl;
  }
};
#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename T>
class Conv3DOp : public BinaryOp<T> {
 public:
  explicit Conv3DOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, 'N') == 1 &&
         GetTensorDim(stride_, data_format_, 'C') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    cudnn_use_autotune_ = CudnnUseAutotune();
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_z, in_y, in_x, in_channels ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_z, filter_y, filter_x, in_channels, out_channels]
    const Tensor& filter = context->input(1);

    // NOTE: The ordering of the spatial dimensions is arbitrary, but has to be
    // kept consistent between input/filter/output.
    OP_REQUIRES(context, input.dims() == 5,
                errors::InvalidArgument("input must be 5-dimensional"));
    OP_REQUIRES(context, filter.dims() == 5,
                errors::InvalidArgument("filter must be 5-dimensional"));

    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    const int64 in_batch = GetTensorDim(input, data_format_, 'N');

    const int64 out_depth = filter.dim_size(4);
    OP_REQUIRES(
        context, in_depth == filter.dim_size(3),
        errors::InvalidArgument("input and filter must have the same depth"));

    // Dimension order for these arrays is: z, y, x.
    std::array<int64, 3> input_size = {
        {GetTensorDim(input, data_format_, '0'),
         GetTensorDim(input, data_format_, '1'),
         GetTensorDim(input, data_format_, '2')}};
    std::array<int64, 3> filter_size = {
        {filter.dim_size(0), filter.dim_size(1), filter.dim_size(2)}};
    std::array<int64, 3> strides = {{GetTensorDim(stride_, data_format_, '0'),
                                     GetTensorDim(stride_, data_format_, '1'),
                                     GetTensorDim(stride_, data_format_, '2')}};
    std::array<int64, 3> out, padding;

    OP_REQUIRES_OK(context, Get3dOutputSize(input_size, filter_size, strides,
                                            padding_, &out, &padding));
    TensorShape out_shape = ShapeFromFormat(
        data_format_, in_batch, {{out[0], out[1], out[2]}}, out_depth);
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // Return early if nothing to do.
    if (out_shape.num_elements() == 0) return;

    LaunchConvOp<Device, T>::launch(context, cudnn_use_autotune_, input, filter,
                                    strides, padding_, data_format_, output);
  }

 private:
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool cudnn_use_autotune_;
};

#define REGISTER_CPU_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv3D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv3DOp<CPUDevice, T>);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA

// A dummy type to group forward convolution autotune results together.
struct Conv3dAutoTuneGroup {
  static string name() { return "Conv3d"; }
};
typedef AutoTuneSingleton<Conv3dAutoTuneGroup, ConvParameters,
                          perftools::gputools::dnn::AlgorithmConfig>
    AutoTuneConv3d;

// TODO(mjanusz): Share logic with 2d implementation as much as possible.
template <typename T>
struct LaunchConvOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, bool cudnn_use_autotune,
                     const Tensor& input_param, const Tensor& filter,
                     const std::array<int64, 3>& strides, const Padding padding,
                     TensorFormat data_format, Tensor* output) {
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    Tensor input = input_param;

    const int64 in_batch = GetTensorDim(input, data_format, 'N');
    int64 in_planes = GetTensorDim(input, data_format, '0');
    int64 in_rows = GetTensorDim(input, data_format, '1');
    int64 in_cols = GetTensorDim(input, data_format, '2');
    const int64 in_depth = GetTensorDim(input, data_format, 'C');

    const int64 filter_planes = filter.dim_size(0);
    const int64 filter_rows = filter.dim_size(1);
    const int64 filter_cols = filter.dim_size(2);
    const int64 out_depth = filter.dim_size(4);

    int64 pad_planes = 0, pad_rows = 0, pad_cols = 0;
    int64 out_planes = GetTensorDim(*output, data_format, '0');
    int64 out_rows = GetTensorDim(*output, data_format, '1');
    int64 out_cols = GetTensorDim(*output, data_format, '2');

    if (padding == Padding::SAME) {
      pad_planes = std::max<int64>(
          0, (out_planes - 1) * strides[0] + filter_planes - in_planes);
      pad_rows = std::max<int64>(
          0, (out_rows - 1) * strides[1] + filter_rows - in_rows);
      pad_cols = std::max<int64>(
          0, (out_cols - 1) * strides[2] + filter_cols - in_cols);
    }

    // NOTE: This only works in NHWC.
    if (filter_planes == 1 && filter_rows == 1 && filter_cols == 1 &&
        strides[0] == 1 && strides[1] == 1 && strides[2] == 1 &&
        data_format == FORMAT_NHWC) {
      // 1x1 filter, so call cublas directly.
      const uint64 m = in_batch * in_planes * in_rows * in_cols;
      const uint64 k = in_depth;
      const uint64 n = out_depth;

      auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                  input.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                  filter.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(output->template flat<T>().data(),
                                  output->template flat<T>().size());

      auto no_transpose = perftools::gputools::blas::Transpose::kNoTranspose;
      bool blas_launch_status =
          stream
              ->ThenBlasGemm(no_transpose, no_transpose, n, m, k, 1.0f, b_ptr,
                             n, a_ptr, k, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        ctx->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                        ", n=", n, ", k=", k));
      }
      return;
    } else if (filter_planes == in_planes && filter_rows == in_rows &&
               filter_cols == in_cols && padding == Padding::VALID &&
               data_format == FORMAT_NHWC) {
      // The input data and filter have the same planes/height/width, so call
      // cublas directly.
      const uint64 m = in_batch;
      const uint64 k = in_planes * in_rows * in_cols * in_depth;
      const uint64 n = out_depth;

      auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                  input.template flat<T>().size());
      auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                  filter.template flat<T>().size());
      auto c_ptr = AsDeviceMemory(output->template flat<T>().data(),
                                  output->template flat<T>().size());

      auto no_transpose = perftools::gputools::blas::Transpose::kNoTranspose;
      bool blas_launch_status =
          stream
              ->ThenBlasGemm(no_transpose, no_transpose, n, m, k, 1.0f, b_ptr,
                             n, a_ptr, k, 0.0f, &c_ptr, n)
              .ok();
      if (!blas_launch_status) {
        ctx->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                        ", n=", n, ", k=", k));
      }
      return;
    }

    if (padding == Padding::SAME) {
      const bool rows_odd = (pad_rows % 2 != 0);
      const bool cols_odd = (pad_cols % 2 != 0);
      const bool planes_odd = (pad_planes % 2 != 0);

      // Necessary because cuDNN only supports symmetric padding.
      // TODO(mjanusz): Consider making this optional? This would save some
      // overhead and would work as long as an op trained this way is only
      // used on GPU.
      if (rows_odd || cols_odd || planes_odd) {
        const int64 new_in_rows = in_rows + rows_odd;
        const int64 new_in_cols = in_cols + cols_odd;
        const int64 new_in_planes = in_planes + planes_odd;

        Tensor transformed_input;
        TensorShape transformed_shape = ShapeFromFormat(
            data_format, in_batch, {{new_in_planes, new_in_rows, new_in_cols}},
            in_depth);
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, transformed_shape,
                                    &transformed_input));

        functor::PadInput<GPUDevice, T, int, 5>()(
            ctx->eigen_device<GPUDevice>(), To32Bit(input_param.tensor<T, 5>()),
            {{0, 0, 0}}, {{planes_odd, rows_odd, cols_odd}},
            To32Bit(transformed_input.tensor<T, 5>()), data_format);
        input = transformed_input;
        in_rows = new_in_rows;
        in_cols = new_in_cols;
        in_planes = new_in_planes;
      }
    }

    if (data_format == FORMAT_NHWC) {
      const TensorShape nchw_shape = ShapeFromFormat(
          FORMAT_NCHW, in_batch, {{in_planes, in_rows, in_cols}}, in_depth);
      if (in_depth > 1) {
        Tensor transformed_input;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                               nchw_shape, &transformed_input));
        // input: [b, x, y, z, d]
        // t_input: [b, d, x, y, z]
        // NCDHW is the only format universally supported by cuDNN.
        functor::NHWCToNCHW<GPUDevice, T, 5>()(
            ctx->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(input).tensor<T, 5>(),
            transformed_input.tensor<T, 5>());
        input = transformed_input;
      } else {
        CHECK(input.CopyFrom(input, nchw_shape));
      }
    }

    CHECK(pad_rows >= 0 && pad_cols >= 0 && pad_planes >= 0)
        << "Negative paddings: (" << pad_rows << ", " << pad_cols << ", "
        << pad_planes << ")";
    perftools::gputools::dnn::BatchDescriptor input_desc(3);
    input_desc.set_count(in_batch)
        .set_feature_map_count(in_depth)
        .set_spatial_dim(DimIndex::X, in_cols)
        .set_spatial_dim(DimIndex::Y, in_rows)
        .set_spatial_dim(DimIndex::Z, in_planes)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::BatchDescriptor output_desc(3);
    output_desc.set_count(in_batch)
        .set_spatial_dim(DimIndex::X, out_cols)
        .set_spatial_dim(DimIndex::Y, out_rows)
        .set_spatial_dim(DimIndex::Z, out_planes)
        .set_feature_map_count(out_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::FilterDescriptor filter_desc(3);
    filter_desc.set_spatial_dim(DimIndex::X, filter_cols)
        .set_spatial_dim(DimIndex::Y, filter_rows)
        .set_spatial_dim(DimIndex::Z, filter_planes)
        .set_input_feature_map_count(in_depth)
        .set_output_feature_map_count(out_depth);
    perftools::gputools::dnn::ConvolutionDescriptor conv_desc(3);
    conv_desc.set_filter_stride(DimIndex::X, strides[2])
        .set_filter_stride(DimIndex::Y, strides[1])
        .set_filter_stride(DimIndex::Z, strides[0])
        .set_zero_padding(DimIndex::X, pad_cols / 2)
        .set_zero_padding(DimIndex::Y, pad_rows / 2)
        .set_zero_padding(DimIndex::Z, pad_planes / 2);

    Tensor transformed_filter;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                TensorShape({out_depth, in_depth, filter_planes,
                                             filter_rows, filter_cols}),
                                &transformed_filter));
    // filter: [x, y, z, in, out]
    // t_filter: [out, in, x, y, z]
    functor::TransformFilter<GPUDevice, T, int, 5>()(
        ctx->eigen_device<GPUDevice>(), To32Bit(filter.tensor<T, 5>()),
        To32Bit(transformed_filter.tensor<T, 5>()));

    Tensor transformed_output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(
                 DataTypeToEnum<T>::value,
                 ShapeFromFormat(FORMAT_NCHW, in_batch,
                                 {{out_planes, out_rows, out_cols}}, out_depth),
                 &transformed_output));

    auto input_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                    input.template flat<T>().size());
    auto filter_ptr =
        AsDeviceMemory(transformed_filter.template flat<T>().data(),
                       transformed_filter.template flat<T>().size());
    auto output_ptr =
        AsDeviceMemory(transformed_output.template flat<T>().data(),
                       transformed_output.template flat<T>().size());

    static int64 ConvolveScratchSize = GetCudnnWorkspaceLimit(
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32);  // 4GB by default

    int device_id = stream->parent()->device_ordinal();
    DataType dtype = input.dtype();
    ConvParameters conv_parameters = {
        in_batch,
        in_depth,
        {{in_planes, in_rows, in_cols}},
        out_depth,
        {{filter_planes, filter_rows, filter_cols}},
        {{strides[0], strides[1], strides[2]}},
        {{pad_planes, pad_rows, pad_cols}},
        dtype,
        device_id,
    };

    using perftools::gputools::dnn::AlgorithmConfig;
    using perftools::gputools::dnn::AlgorithmType;
    using perftools::gputools::dnn::ProfileResult;
    using perftools::gputools::dnn::kDefaultAlgorithm;

    AlgorithmConfig algorithm_config;

    if (cudnn_use_autotune && !AutoTuneConv3d::GetInstance()->Find(
                                  conv_parameters, &algorithm_config)) {
      std::vector<AlgorithmType> algorithms;
      CHECK(stream->parent()->GetConvolveAlgorithms(
          conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms));
      ProfileResult best_result;
      ProfileResult best_result_no_scratch;
      for (auto profile_algorithm : algorithms) {
        // TODO(zhengxq): profile each algorithm multiple times to better
        // accuracy.
        CudnnScratchAllocator scratch_allocator(ConvolveScratchSize, ctx);
        ProfileResult profile_result;
        bool cudnn_launch_status =
            stream
                ->ThenConvolveWithAlgorithm(
                    input_desc, input_ptr, filter_desc, filter_ptr, conv_desc,
                    output_desc, &output_ptr, &scratch_allocator,
                    AlgorithmConfig(profile_algorithm), &profile_result)
                .ok();
        if (cudnn_launch_status) {
          if (profile_result.is_valid()) {
            if (profile_result.elapsed_time_in_ms() <
                best_result.elapsed_time_in_ms()) {
              best_result = profile_result;
            }
            if (scratch_allocator.TotalByteSize() == 0 &&
                profile_result.elapsed_time_in_ms() <
                    best_result_no_scratch.elapsed_time_in_ms()) {
              best_result_no_scratch = profile_result;
            }
          }
        }
      }
      OP_REQUIRES(ctx,
                  best_result.is_valid() &&
                      best_result.algorithm() != kDefaultAlgorithm,
                  errors::NotFound("No algorithm worked!"));
      OP_REQUIRES(ctx,
                  best_result_no_scratch.is_valid() &&
                      best_result_no_scratch.algorithm() != kDefaultAlgorithm,
                  errors::NotFound("No algorithm without scratch worked!"));
      algorithm_config.set_algorithm(best_result.algorithm());
      algorithm_config.set_algorithm_no_scratch(
          best_result_no_scratch.algorithm());
      AutoTuneConv3d::GetInstance()->Insert(conv_parameters, algorithm_config);
    }

    CudnnScratchAllocator scratch_allocator(ConvolveScratchSize, ctx);
    bool cudnn_launch_status =
        stream
            ->ThenConvolveWithAlgorithm(input_desc, input_ptr, filter_desc,
                                        filter_ptr, conv_desc, output_desc,
                                        &output_ptr, &scratch_allocator,
                                        algorithm_config, nullptr)
            .ok();

    if (!cudnn_launch_status) {
      ctx->SetStatus(errors::Internal(
          "cuDNN launch failure : input shape(", input.shape().DebugString(),
          ") filter shape(", filter.shape().DebugString(), ")"));
    }

    if (data_format == FORMAT_NHWC) {
      // t_output: [b, out, x, y, z]
      // output: [b, x, y, z, out]
      functor::NCHWToNHWC<GPUDevice, T, 5>()(
          ctx->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(transformed_output).tensor<T, 5>(),
          output->tensor<T, 5>());
    } else {
      *output = transformed_output;
    }
  }
};

// Forward declarations of the functor specializations for GPU.
// This ensures that the custom implementation is used instead of the default
// Eigen one (which is used for CPU).
namespace functor {
#define DECLARE_GPU_SPEC(T)                                           \
  template <>                                                         \
  void TransformFilter<GPUDevice, T, int, 5>::operator()(             \
      const GPUDevice& d, typename TTypes<T, 5, int>::ConstTensor in, \
      typename TTypes<T, 5, int>::Tensor out);                        \
  template <>                                                         \
  void ReverseTransformFilter<GPUDevice, T, 5>::operator()(           \
      const GPUDevice& d, typename TTypes<T, 5>::ConstTensor in,      \
      typename TTypes<T, 5>::Tensor out);                             \
  template <>                                                         \
  void PadInput<GPUDevice, T, int, 5>::operator()(                    \
      const GPUDevice& d, typename TTypes<T, 5, int>::ConstTensor in, \
      const std::array<int, 3>& padding_left,                         \
      const std::array<int, 3>& padding_right,                        \
      typename TTypes<T, 5, int>::Tensor out, TensorFormat format);

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC

}  // namespace functor

// Registration of the GPU implementations.
REGISTER_KERNEL_BUILDER(
    Name("Conv3D").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    Conv3DOp<GPUDevice, float>);
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(
    Name("Conv3D").Device(DEVICE_SYCL).TypeConstraint<float>("T"),
    Conv3DOp<SYCLDevice, float>);
#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
