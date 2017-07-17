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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/depthwise_conv_op.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

// In depthwise convolution, one input is convolved into depth_multipler
// outputs and the outputs don't need to be reduced again like what regular
// convolution does.
//  However, the way to apply filters to inputs is exactly the same as the
// regular convolution. Please refer to the regular convolution kernels for
// more details.

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename T>
struct LaunchDepthwiseConvOp;

// Computes the vectorized product of 'input_buffer' and 'filter' and stores
// result in 'output' at location specified by 'out_r' and 'out_c'.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//   Both 'input_buffer' and 'filter' are padded to register-width boundaries.
//
//   input_buffer [rows, cols, in_depth, depth_multiplier]
//     [a0, a0, a1, a1] [a2, a2, 0, 0] [b0, b0, b1, b1] [b2, b2, 0, 0]
//     [e0, e0, e1, e1] [e2, e2, 0, 0] [f0, f0, f1, f1] [f2, f2, 0, 0]
//
//   filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, 0, 0] [u1, v1, w1, x1] [y1, z1, 0, 0]
//     [u2, v2, w2, x2] [y2, z2, 0, 0] [u3, v3, w3, x3] [y3, z3, 0, 0]
//
//   First output register [in_depth, depth_multiplier]
//     [q0, q1, q2, q3] = ([a0, a0, a1, a1] x [u0, v0, w0, x0]) +
//                        ([b0, b0, b1, b1] x [u1, v1, w1, x1]) +
//                        ([e0, e0, e1, e1] x [u2, v2, w2, x2]) +
//                        ([f0, f0, f1, f1] x [u3, v3, w3, x3])
//
// TODO(andydavis) Experiment with processing multiple inputs per input buffer.
template <typename T>
struct DepthwiseConv2DKernel {
  static void Run(const DepthwiseArgs& args,
                  const int64 padded_filter_inner_dim_size, const int64 out_r,
                  const int64 out_c, const T* filter, const T* input_buffer,
                  T* output, TensorFormat data_format) {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    const int64 out_depth = args.out_depth;
    const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
    const int64 output_scalar_size = out_depth % kPacketSize;
    const int64 output_vectorized_size =
        (out_depth / kPacketSize) * kPacketSize;
    const int64 base_output_index = (out_r * args.out_cols + out_c) * out_depth;

    for (int i = 0; i < output_vectorized_size; i += kPacketSize) {
      // Reset accumulator.
      auto vaccum = Eigen::internal::pset1<Packet>(0);
      for (int j = 0; j < filter_spatial_size; ++j) {
        // Calculate index.
        const int64 index = i + j * padded_filter_inner_dim_size;
        // Load filter.
        // TODO(andydavis) Unroll 'out_c' loop in caller so we can load
        // multiple inputs here to amortize the cost of each filter block load.
        const auto filter_block =
            Eigen::internal::ploadu<Packet>(filter + index);
        // Load input.
        const auto data_block =
            Eigen::internal::ploadu<Packet>(input_buffer + index);
        // Vector multiply-add.
        vaccum =
            Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);
      }
      // Store vector accumulator to output.
      Eigen::internal::pstoreu<T>(output + base_output_index + i, vaccum);
    }

    if (output_scalar_size > 0) {
      auto vaccum = Eigen::internal::pset1<Packet>(0);
      for (int j = 0; j < filter_spatial_size; ++j) {
        const int64 index =
            output_vectorized_size + j * padded_filter_inner_dim_size;
        const auto filter_block =
            Eigen::internal::ploadu<Packet>(filter + index);
        const auto data_block =
            Eigen::internal::ploadu<Packet>(input_buffer + index);
        vaccum =
            Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);
      }
      // Load accumulator into an array and loop through output.
      T out_buf[kPacketSize];
      Eigen::internal::pstoreu<T>(out_buf, vaccum);
      const int64 last_output_index =
          base_output_index + output_vectorized_size;
      for (int j = 0; j < output_scalar_size; ++j) {
        output[last_output_index + j] = out_buf[j];
      }
    }
  }
};

// Computes the depthwise conv2d of 'input' by 'depthwise_filter' and stores
// the result in 'output'. This implementation trades off copying small patches
// of the input to achieve better data alignment, which enables vectorized
// load/store and multiply-add operations (see comments at InputBufferCopyOp and
// DepthwiseConv2DKernel for details).
//
// TODO(andydavis) Evaluate the performance of processing multiple input
// patches in the inner loop.
// TODO(andydavis) Consider a zero-copy implementation for the case when
// 'in_depth' is a multiple of register width, and 'depth_multipler' is one.
// TODO(andydavis) Evaluate the performance of alternative implementations.
template <typename T>
struct LaunchDepthwiseConvOp<CPUDevice, T> {
  typedef typename Eigen::internal::packet_traits<T>::type Packet;

  static void launch(OpKernelContext* ctx, const DepthwiseArgs& args,
                     const T* input, const T* depthwise_filter, T* output,
                     TensorFormat data_format) {
    OP_REQUIRES(
        ctx, data_format == FORMAT_NHWC,
        errors::Unimplemented(
            "Depthwise convolution on CPU is only supported for NHWC format"));
    static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));

    // Pad 'depthwise_filter' to vector register width (if needed).
    const bool pad_filter = (args.out_depth % kPacketSize) == 0 ? false : true;
    Tensor padded_filter;
    if (pad_filter) {
      // Allocate space for padded filter.
      const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
      const int64 padded_filter_inner_dim_size =
          ((args.out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  TensorShape({filter_spatial_size,
                                               padded_filter_inner_dim_size}),
                                  &padded_filter));
      // Write out padded filter.
      functor::DepthwiseFilterPadOp<T>()(
          args, depthwise_filter, padded_filter.template flat<T>().data());
    }
    const T* filter_data =
        pad_filter ? padded_filter.template flat<T>().data() : depthwise_filter;

    // Computes one shard of depthwise conv2d output.
    auto shard = [&ctx, &args, &input, &filter_data, &output, data_format](
                     int64 start, int64 limit) {
      static const int64 kPacketSize = (sizeof(Packet) / sizeof(T));
      const int64 input_image_size =
          args.in_rows * args.in_cols * args.in_depth;
      const int64 output_image_size =
          args.out_rows * args.out_cols * args.out_depth;
      const int64 filter_spatial_size = args.filter_rows * args.filter_cols;
      const int64 padded_filter_inner_dim_size =
          ((args.out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;

      // Allocate buffer for local input regions.
      Tensor input_buffer;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  TensorShape({filter_spatial_size,
                                               padded_filter_inner_dim_size}),
                                  &input_buffer));
      T* input_buffer_data = input_buffer.template flat<T>().data();

      for (int64 i = start; i < limit; ++i) {
        const int64 b = i / args.out_rows;
        const int64 in_base = b * input_image_size;
        const int64 out_base = b * output_image_size;

        const int64 out_r = i % args.out_rows;

        for (int64 out_c = 0; out_c < args.out_cols; ++out_c) {
          // Populate 'input_buffer_data' with data from local input region.
          functor::DepthwiseInputCopyOp<T>()(args, padded_filter_inner_dim_size,
                                             out_r, out_c, input + in_base,
                                             input_buffer_data);

          // Process buffered input across all filters and store to output.
          DepthwiseConv2DKernel<T>::Run(
              args, padded_filter_inner_dim_size, out_r, out_c, filter_data,
              input_buffer_data, output + out_base, data_format);
        }
      }
    };

    const int64 total_shards = args.batch * args.out_rows;

    // Empirically tested to give reasonable performance boosts at batch size 1
    // without reducing throughput at batch size 32.
    const float kCostMultiplier = 2.5f;

    // TODO(andydavis): Estimate shard cost (in cycles) based on the number of
    // flops/loads/stores required to compute one shard.
    const int64 shard_cost = kCostMultiplier * args.out_cols * args.out_depth;

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, total_shards,
          shard_cost, shard);
  }
};

// Extern template instantiated in conv_ops.cc.
extern template class LaunchConv2DOp<CPUDevice, float>;

#if GOOGLE_CUDA

template <typename T>
struct DepthwiseConv2dGPULaunch {
  static void Run(const GPUDevice& d, const DepthwiseArgs args, const T* input,
                  const T* filter, T* output, TensorFormat data_format);
};

template <typename T>
struct LaunchDepthwiseConvOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, const DepthwiseArgs args,
                     const T* input, const T* filter, T* output,
                     TensorFormat data_format) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    DepthwiseConv2dGPULaunch<T>().Run(d, args, input, filter, output,
                                      data_format);
    auto stream = ctx->op_device_context()->stream();
    OP_REQUIRES(
        ctx, stream->ok(),
        errors::Internal(
            "Launch of gpu kernel for DepthwiseConv2dGPULaunch failed"));
  }
};

// Extern template instantiated in conv_ops.cc.
extern template class LaunchConv2DOp<GPUDevice, float>;

#endif

template <typename Device, typename T>
class DepthwiseConv2dNativeOp : public BinaryOp<T> {
 public:
  explicit DepthwiseConv2dNativeOp(OpKernelConstruction* context)
      : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    stride_ = GetTensorDim(strides_, data_format_, 'H');
    const int64 stride_w = GetTensorDim(strides_, data_format_, 'W');
    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');

    OP_REQUIRES(context, stride_ == stride_w,
                errors::InvalidArgument(
                    "Current implementation only supports equal length "
                    "strides in the row and column dimensions."));
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));

    // For special case when in_depth == 1.
    use_cudnn_ = CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, depth_multiplier]
    const Tensor& filter = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    // in_depth for input and filter must match.
    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(
        context, in_depth == filter.dim_size(2),
        errors::InvalidArgument("input and filter must have the same depth: ",
                                in_depth, " vs ", filter.dim_size(2)));

    // The last dimension for filter is depth multiplier.
    const int32 depth_multiplier = filter.dim_size(3);

    // The output depth is input depth x depth multipler
    const int32 out_depth = in_depth * depth_multiplier;

    const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int32 input_rows = static_cast<int32>(input_rows_raw);
    const int32 filter_rows = filter.dim_size(0);

    const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int32 input_cols = static_cast<int32>(input_cols_raw);
    const int32 filter_cols = filter.dim_size(1);

    // The first dimension for input is batch.
    const int32 batch = input.dim_size(0);

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_,
                                         padding_, &out_cols, &pad_cols));
    TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);
    OP_REQUIRES(
        context, out_shape.num_elements() <= 2147483647,
        errors::InvalidArgument("total number of outputs should be within the "
                                "range of int which is used in the GPU kernel",
                                in_depth, " vs ", filter.dim_size(2)));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "DepthwiseConv2dNative: "
            << " Input: [" << batch << ", " << input_rows << ", " << input_cols
            << ", " << in_depth << "]; Filter: [" << filter_rows << ", "
            << filter_cols << ", " << in_depth << ", " << depth_multiplier
            << "]; stride = " << stride_ << ", pad_rows = " << pad_rows
            << ", pad_cols = " << pad_cols << ", output: [" << batch << ", "
            << out_rows << ", " << out_cols << ", " << out_depth << "]";

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    // If in_depth==1, this operation is just a standard convolution, so
    // invoke that op.
    if (std::is_same<T, float>::value && in_depth == 1) {
      launcher_.launch(context, use_cudnn_, cudnn_use_autotune_, input, filter,
                       stride_, stride_, BrainPadding2EigenPadding(padding_),
                       output, data_format_);
      return;
    }

    DepthwiseArgs args;
    args.batch = batch;
    args.in_rows = input_rows;
    args.in_cols = input_cols;
    args.in_depth = in_depth;
    args.filter_rows = filter_rows;
    args.filter_cols = filter_cols;
    args.depth_multiplier = depth_multiplier;
    args.stride = stride_;
    args.pad_rows = pad_rows;
    args.pad_cols = pad_cols;
    args.out_rows = out_rows;
    args.out_cols = out_cols;
    args.out_depth = out_depth;

    auto input_ptr = input.template flat<T>().data();
    auto filter_ptr = filter.template flat<T>().data();
    auto output_ptr = output->template flat<T>().data();
    LaunchDepthwiseConvOp<Device, T>::launch(
        context, args, input_ptr, filter_ptr, output_ptr, data_format_);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  int64 stride_;  // in height/width dimension.

  // For the case in_depth == 1.
  LaunchConv2DOp<Device, T> launcher_;
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(DepthwiseConv2dNativeOp);
};

#ifdef TENSORFLOW_USE_SYCL
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
class DepthwiseConv2dSYCLKernelNHWC {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;
public:
  DepthwiseConv2dSYCLKernelNHWC(const DepthwiseArgs args,
                                read_accessor input_data_accessor,
                                read_accessor filter_data_accessor,
                                write_accessor output_data_accessor)
                                : args_(args),
                                input_data_accessor_(input_data_accessor),
                                filter_data_accessor_(filter_data_accessor),
                                output_data_accessor_(output_data_accessor){}
  void operator()(cl::sycl::item<1> item){
    T* input_data = ConvertToActualTypeSycl(T, input_data_accessor_);
    T* filter_data = ConvertToActualTypeSycl(T, filter_data_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_data_accessor_);

    const int thread_id = item.get_linear_id();

    const int filter_rows =
        kKnownFilterHeight < 0 ? args_.filter_rows : kKnownFilterHeight;
    const int filter_cols =
        kKnownFilterWidth < 0 ? args_.filter_cols : kKnownFilterWidth;
    const int depth_multiplier =
        kKnownDepthMultiplier < 0 ? args_.depth_multiplier : kKnownDepthMultiplier;

    // Compute the indexes of this thread in the output.
    const int OD = thread_id % args_.out_depth;
    const int OC = (thread_id / args_.out_depth) % args_.out_cols;
    const int OR = (thread_id / args_.out_depth / args_.out_cols) % args_.out_rows;
    const int OB = thread_id / args_.out_depth / args_.out_cols / args_.out_rows;
    // Compute the input depth and the index of depth multiplier.
    const int in_d = OD / depth_multiplier;
    const int multiplier = OD % depth_multiplier;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int input_row_start = OR * args_.stride - args_.pad_rows;
    const int input_col_start = OC * args_.stride - args_.pad_cols;
    const int input_row_end = input_row_start + filter_rows;
    const int input_col_end = input_col_start + filter_cols;

    T sum = T(0);

    const int input_offset_temp = args_.in_rows * OB;
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < args_.in_rows && input_col_end < args_.in_cols) {
      for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const int filter_offset_temp = filter_cols * f_r;
        for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;

          const int input_offset =
              in_d + args_.in_depth * (in_c + args_.in_cols * (in_r + input_offset_temp));
          const int filter_offset =
              multiplier +
              depth_multiplier * (in_d + args_.in_depth * (f_c + filter_offset_temp));
          sum += input_data[input_offset] * filter_data[filter_offset];
        }
      }
    } else {
      for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const int filter_offset_temp = filter_cols * f_r;
        for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;
          if (in_r >= 0 && in_r < args_.in_rows && in_c >= 0 && in_c < args_.in_cols) {
            const int in_c = input_col_start + f_c;

            const int input_offset =
                in_d + args_.in_depth * (in_c + args_.in_cols * (in_r + input_offset_temp));
            const int filter_offset =
                multiplier + depth_multiplier *
                                 (in_d + args_.in_depth * (f_c + filter_offset_temp));
            sum += input_data[input_offset] * filter_data[filter_offset];
          }
        }
      }
    }
    output_data[thread_id] = sum;
  }
private:
  const DepthwiseArgs args_;
  const read_accessor input_data_accessor_;
  const read_accessor filter_data_accessor_;
  write_accessor output_data_accessor_;
};

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
class DepthwiseConv2dSYCLKernelNCHW {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;
public:
  DepthwiseConv2dSYCLKernelNCHW(const DepthwiseArgs args,
                                read_accessor input_data_accessor,
                                read_accessor filter_data_accessor,
                                write_accessor output_data_accessor)
                                : args_(args),
                                input_data_accessor_(input_data_accessor),
                                filter_data_accessor_(filter_data_accessor),
                                output_data_accessor_(output_data_accessor){}
  void operator()(cl::sycl::item<1> item){
    T* input_data = ConvertToActualTypeSycl(T, input_data_accessor_);
    T* filter_data = ConvertToActualTypeSycl(T, filter_data_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_data_accessor_);

    const int thread_id = item.get_linear_id();

    const int filter_rows =
        kKnownFilterHeight < 0 ? args_.filter_rows : kKnownFilterHeight;
    const int filter_cols =
        kKnownFilterWidth < 0 ? args_.filter_cols : kKnownFilterWidth;
    const int depth_multiplier =
        kKnownDepthMultiplier < 0 ? args_.depth_multiplier : kKnownDepthMultiplier;
    // Compute the indexes of this thread in the output.
    //
    // We want coalesced reads so we make sure that each warp reads
    // a contiguous chunk of memory.
    //
    // THIS IS PROBABLY WRONG, we are not doing coalesced reads
    // into the input, because of the depth multiplier division...
    const int OC = thread_id % args_.out_cols;
    const int OR = (thread_id / args_.out_cols) % args_.out_rows;
    const int OD = (thread_id / args_.out_cols / args_.out_rows) % args_.out_depth;
    const int OB = thread_id / args_.out_cols / args_.out_rows / args_.out_depth;

    // Compute the input depth and the index of depth multiplier
    // based off the output depth index that this thread is
    // computing n.
    const int in_d = OD / depth_multiplier;
    const int multiplier = OD % depth_multiplier;

    // Data is stored in the following format (let's assume we
    // flatten the height and width into one contiguous dimension
    // called "P".
    //
    // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
    //
    // Each row contains in_depth * in_rows * in_cols values
    // for each sample in the batch.
    //
    // We can further flatten it into:
    //
    // B1C1P1 B1C1P2 .....
    // B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 .....
    // B2C2P1 B2C2P2 ....
    //
    // where each row is a contiguous array of all of the spatial
    // pixels for a given batch and input depth.  The following
    // loop unrolls across the filter dimensions for a given thread,
    // indexing into the filter value and the corresponding input
    // patch.
    //
    // We can compute the index into the patch once right here.
    const int input_offset_temp = (OB * args_.in_depth + in_d) * (args_.in_rows * args_.in_cols);

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    //
    // We perform an additional optimization, where we can determine
    // whether the patch fits within the image indices statically, and
    // avoid boundary checking within the loop.
    const int input_row_start = OR * args_.stride - args_.pad_rows;
    const int input_col_start = OC * args_.stride - args_.pad_cols;
    const int input_row_end = input_row_start + filter_rows;
    const int input_col_end = input_col_start + filter_cols;

    T sum = T(0);
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < args_.in_rows && input_col_end < args_.in_cols) {
      // Loop that doesn't need to check for boundary conditions.
      for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const int filter_offset_temp = filter_cols * f_r;
        for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;

          const int input_offset =
              (input_offset_temp) + (in_r * args_.in_cols) + in_c;
          const int filter_offset =
              multiplier +
              depth_multiplier * (in_d + args_.in_depth * (f_c + filter_offset_temp));
          sum += input_data[input_offset] * filter_data[filter_offset];
        }
      }
    } else {
      // Loop that needs to check for boundary conditions.
      for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const int filter_offset_temp = filter_cols * f_r;
        for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;
          // TODO(vrv): the in_r check can be done outside of this loop;
          // benchmark both methods to determine the better decision.
          if (in_r >= 0 && in_r < args_.in_rows && in_c >= 0 && in_c < args_.in_cols) {
            const int in_c = input_col_start + f_c;

            // input_offset_temp indexes into the start of memory
            // where the spatial data starts.
            const int input_offset =
                (input_offset_temp) + (in_r * args_.in_cols) + in_c;

            const int filter_offset =
                multiplier + depth_multiplier *
                                 (in_d + args_.in_depth * (f_c + filter_offset_temp));
            sum += input_data[input_offset] * filter_data[filter_offset];
          }
        }
      }
    }
    output_data[thread_id] = sum;
  }
private:
  const DepthwiseArgs args_;
  const read_accessor input_data_accessor_;
  const read_accessor filter_data_accessor_;
  write_accessor output_data_accessor_;
};

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
void LaunchDepthwiseConv2dSYCL(const SYCLDevice& d, const DepthwiseArgs args,
                              const Tensor& input, const Tensor& filter, Tensor* output,
                              TensorFormat data_format) {
  const int num_threads = output->NumElements();

  auto input_data_buffer = d.get_sycl_buffer(input.template flat<T>().data());
  auto filter_data_buffer = d.get_sycl_buffer(filter.template flat<T>().data());
  auto output_data_buffer = d.get_sycl_buffer(output->template flat<T>().data());

  d.sycl_queue().submit([&](cl::sycl::handler& cgh) {
    auto input_data_access =
        input_data_buffer
            .template get_access<cl::sycl::access::mode::read>(cgh);
    auto filter_data_access =
        filter_data_buffer
            .template get_access<cl::sycl::access::mode::read>(cgh);
    auto output_data_access =
        output_data_buffer
            .template get_access<cl::sycl::access::mode::write>(cgh);

    if(data_format == FORMAT_NHWC){
      DepthwiseConv2dSYCLKernelNHWC<T, kKnownFilterWidth, kKnownFilterHeight,
                                    kKnownDepthMultiplier> functor(
          args, input_data_access, filter_data_access, output_data_access);
      cgh.parallel_for(cl::sycl::range<1>(num_threads), functor);
    } else if (data_format == FORMAT_NCHW) {
      DepthwiseConv2dSYCLKernelNCHW<T, kKnownFilterWidth, kKnownFilterHeight,
                                    kKnownDepthMultiplier> functor(
          args, input_data_access, filter_data_access, output_data_access);
      cgh.parallel_for(cl::sycl::range<1>(num_threads), functor);
    } else {
      assert(false && "Incorrect data format");
      return;
    }
  });
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
void LaunchDepthwiseConv2dSYCL(const SYCLDevice& d, const DepthwiseArgs args,
                              const Tensor& input, const Tensor& filter,
                              Tensor* output, TensorFormat data_format) {
  if (args.depth_multiplier == 1) {
    LaunchDepthwiseConv2dSYCL<T, kKnownFilterWidth, kKnownFilterHeight, 1>(
        d, args, input, filter, output, data_format);
  } else {
    LaunchDepthwiseConv2dSYCL<T, kKnownFilterWidth, kKnownFilterHeight, -1>(
        d, args, input, filter, output, data_format);
  }
}

template <typename T>
struct LaunchDepthwiseConvOp<SYCLDevice, T> {
  static void launch(OpKernelContext* ctx, const DepthwiseArgs args,
                     const Tensor& input, const Tensor& filter, Tensor* output,
                     TensorFormat data_format) {
    const SYCLDevice& d = ctx->eigen_device<SYCLDevice>();
    if (args.filter_rows == 3 && args.filter_cols == 3) {
      LaunchDepthwiseConv2dSYCL<T, 3, 3>(d, args, input, filter, output,
                                        data_format);
    } else {
      LaunchDepthwiseConv2dSYCL<T, -1, -1>(d, args, input, filter, output,
                                          data_format);
    }
  }
};

// Extern template instantiated in conv_ops.cc.
extern template class LaunchConv2DOp<SYCLDevice, float>;

template <typename T>
class DepthwiseConv2dNativeOp<SYCLDevice, T> : public BinaryOp<T> {
 public:
   explicit DepthwiseConv2dNativeOp(OpKernelConstruction* context)
       : BinaryOp<T>(context) {
     OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
     string data_format;
     OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
     OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                 errors::InvalidArgument("Invalid data format"));

     OP_REQUIRES(context, strides_.size() == 4,
                 errors::InvalidArgument("Sliding window strides field must "
                                         "specify 4 dimensions"));
     stride_ = GetTensorDim(strides_, data_format_, 'H');
     const int64 stride_w = GetTensorDim(strides_, data_format_, 'W');
     const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
     const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');

     OP_REQUIRES(context, stride_ == stride_w,
                 errors::InvalidArgument(
                     "Current implementation only supports equal length "
                     "strides in the row and column dimensions."));
     OP_REQUIRES(
         context, (stride_n == 1 && stride_c == 1),
         errors::InvalidArgument("Current implementation does not yet support "
                                 "strides in the batch and depth dimensions."));
     OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));

     // For special case when in_depth == 1.
     use_cudnn_ = CanUseCudnn();
     cudnn_use_autotune_ = CudnnUseAutotune();
   }
  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, depth_multiplier]
    const Tensor& filter = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    // in_depth for input and filter must match.
    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(
        context, in_depth == filter.dim_size(2),
        errors::InvalidArgument("input and filter must have the same depth: ",
                                in_depth, " vs ", filter.dim_size(2)));

    // The last dimension for filter is depth multiplier.
    const int32 depth_multiplier = filter.dim_size(3);

    // The output depth is input depth x depth multipler
    const int32 out_depth = in_depth * depth_multiplier;

    const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int32 input_rows = static_cast<int32>(input_rows_raw);
    const int32 filter_rows = filter.dim_size(0);

    const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int32>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int32 input_cols = static_cast<int32>(input_cols_raw);
    const int32 filter_cols = filter.dim_size(1);

    // The first dimension for input is batch.
    const int32 batch = input.dim_size(0);

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_,
                                         padding_, &out_cols, &pad_cols));
    TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);
    OP_REQUIRES(
        context, out_shape.num_elements() <= 2147483647,
        errors::InvalidArgument("total number of outputs should be within the "
                                "range of int which is used in the SYCL kernel",
                                in_depth, " vs ", filter.dim_size(2)));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "DepthwiseConv2dNative: "
            << " Input: [" << batch << ", " << input_rows << ", " << input_cols
            << ", " << in_depth << "]; Filter: [" << filter_rows << ", "
            << filter_cols << ", " << in_depth << ", " << depth_multiplier
            << "]; stride = " << stride_ << ", pad_rows = " << pad_rows
            << ", pad_cols = " << pad_cols << ", output: [" << batch << ", "
            << out_rows << ", " << out_cols << ", " << out_depth << "]";

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    // If in_depth==1, this operation is just a standard convolution, so
    // invoke that op.
    if (std::is_same<T, float>::value && in_depth == 1) {
      launcher_.launch(context, use_cudnn_, cudnn_use_autotune_, input, filter,
                       stride_, stride_, BrainPadding2EigenPadding(padding_),
                       output, data_format_);
      return;
    }

    DepthwiseArgs args;
    args.batch = batch;
    args.in_rows = input_rows;
    args.in_cols = input_cols;
    args.in_depth = in_depth;
    args.filter_rows = filter_rows;
    args.filter_cols = filter_cols;
    args.depth_multiplier = depth_multiplier;
    args.stride = stride_;
    args.pad_rows = pad_rows;
    args.pad_cols = pad_cols;
    args.out_rows = out_rows;
    args.out_cols = out_cols;
    args.out_depth = out_depth;

    LaunchDepthwiseConvOp<SYCLDevice, T>::launch(
        context, args, input, filter, output, data_format_);
  }

 private:
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;

  int64 stride_;  // in height/width dimension.

  // For the case in_depth == 1.
  LaunchConv2DOp<SYCLDevice, T> launcher_;
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(DepthwiseConv2dNativeOp);
};

REGISTER_KERNEL_BUILDER(Name("DepthwiseConv2dNative")
                            .Device(DEVICE_SYCL).TypeConstraint<float>("T"),
                        DepthwiseConv2dNativeOp<SYCLDevice, float>);

REGISTER_KERNEL_BUILDER(Name("DepthwiseConv2dNative")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<double>("T"),
                        DepthwiseConv2dNativeOp<SYCLDevice, double>);
#endif  // TENSORFLOW_USE_SYCL

#define REGISTER_CPU_KERNEL(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("DepthwiseConv2dNative").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DepthwiseConv2dNativeOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU_KERNEL);
#if !defined(PLATFORM_WINDOWS) || !defined(_DEBUG)
TF_CALL_double(REGISTER_CPU_KERNEL);
#endif

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("DepthwiseConv2dNative").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    DepthwiseConv2dNativeOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("DepthwiseConv2dNative")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T"),
                        DepthwiseConv2dNativeOp<GPUDevice, double>);
#endif

}  // namespace tensorflow
