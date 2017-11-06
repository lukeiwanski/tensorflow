/*
 * Copyright 2017 John Lawson, Codeplay Software.
 * All rights reserved.
 */
#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_grad_ops.h"

#include "tensorflow/core/kernels/conv_ops_im2col_sycl.h"
#include "tensorflow/core/kernels/conv_ops_naive_sycl.h"
#include "tensorflow/core/kernels/conv_ops_sycl_common.h"
#include "tensorflow/core/kernels/conv_ops_winograd_sycl.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
// Forward declarations needed for later specializations.
template <typename Device, typename T>
struct LaunchConv2DOp;

template <typename T>
struct LaunchConv2DOp<SYCLDevice, T> {
  void operator()(OpKernelContext* context, bool use_cudnn,
                  bool cudnn_use_autotune, const Tensor& input,
                  const Tensor& filter, int64 stride_rows, int64 stride_cols,
                  const Padding& padding, Tensor* output,
                  TensorFormat data_format) {
    CHECK(data_format == FORMAT_NHWC) << "SYCL convolution implementation only "
                                         "supports NHWC tensor format.";
    if (launch_matmul(context, input, filter, stride_rows, stride_cols, padding,
                      output)) {
      return;
    }
    const int64 batch = GetTensorDim(input, data_format, 'N');
    const int64 input_rows = GetTensorDim(input, data_format, 'H');
    const int64 input_cols = GetTensorDim(input, data_format, 'W');

    const int64 filter_rows = filter.dim_size(0);
    const int64 filter_cols = filter.dim_size(1);
    const int64 in_depth = filter.dim_size(2);
    const int64 out_depth = filter.dim_size(3);

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding, &out_cols, &pad_cols));

    SYCLConv2DParams params{in_depth,    out_depth,   batch,       input_rows,
                            input_cols,  filter_rows, filter_cols, stride_rows,
                            stride_cols, out_rows,    out_cols,    pad_rows,
                            pad_cols};
    if (stride_rows == 1 && stride_cols == 1 && filter_rows == 3 &&
        filter_cols == 1) {
      LaunchMatmulWinograd<T, 2, 1, 3, 1, ConvType::Forward>::launch(
          context, output, input, filter, params);
      return;
    }
    if (stride_rows == 1 && stride_cols == 1 && filter_rows == 1 &&
        filter_cols == 3) {
      LaunchMatmulWinograd<T, 1, 2, 1, 3, ConvType::Forward>::launch(
          context, output, input, filter, params);
      return;
    }
    if (stride_rows == 1 && stride_cols == 1 && filter_rows == 3 &&
        filter_cols == 3) {
      LaunchMatmulWinograd<T, 2, 2, 3, 3, ConvType::Forward>::launch(
          context, output, input, filter, params);
      return;
    }
    LaunchIm2Col<T, ConvType::Forward>::launch(context, output, input, filter,
                                               params);

    // LaunchConv2DSYCL<T>::launch(context, output, input, filter, params);
  }

 private:
  /**
   * Check whether the convolution can be computed with a single matrix
   * multiply. This is the case when the filter is 1x1 or where the filter is
   * the same size as the input tensor.
   *
   * The MatMulConvFunctor used here is defined in
   * tensorflow/core/kernels/conv_2d.h and just calls Eigen contract.
   *
   * Returns true if the convolution has been launched, and false if the
   * convolution cannot be computed using a matrix multiply.
   */
  bool launch_matmul(OpKernelContext* context, const Tensor& input,
                     const Tensor& filter, int64 stride_rows, int64 stride_cols,
                     const Padding& padding, Tensor* output) {
    if (filter.dim_size(0) == 1 && filter.dim_size(1) == 1 &&
        stride_rows == 1 && stride_cols == 1) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      int conv_width = 1;
      for (int i = 0; i < 3; ++i) {
        conv_width *= output->dim_size(i);
      }

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<SYCLDevice, T>()(
          context->eigen_device<SYCLDevice>(),
          output->shaped<T, 2>({conv_width, filter.dim_size(3)}),
          input.shaped<T, 2>({conv_width, filter.dim_size(2)}),
          filter.shaped<T, 2>({filter.dim_size(2), filter.dim_size(3)}),
          dim_pair);
      return true;
    } else if (filter.dim_size(0) == input.dim_size(1) &&
               filter.dim_size(1) == input.dim_size(2) && padding == VALID) {
      // If the input data and filter have the same height/width,
      // the 2D convolution is reduced to matrix multiplication.
      const int k =
          filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<SYCLDevice, T>()(
          context->eigen_device<SYCLDevice>(),
          output->shaped<T, 2>({input.dim_size(0), filter.dim_size(3)}),
          input.shaped<T, 2>({input.dim_size(0), k}),
          filter.shaped<T, 2>({k, filter.dim_size(3)}), dim_pair);
      return true;
    } else {
      return false;
    }
  }
};
template <typename T>
struct LaunchConv2DBackpropInputOp<SYCLDevice, T> {
  void operator()(OpKernelContext* context, bool /*use_cudnn*/,
                  bool /*cudnn_use_autotune*/, const Tensor& out_backprop,
                  const Tensor& filter, int64 stride_rows, int64 stride_cols,
                  const Padding& padding, Tensor* in_backprop,
                  TensorFormat data_format) {
    if (launch_matmul(context, out_backprop, filter, stride_rows, stride_cols,
                      padding, in_backprop)) {
      return;
    }
    const int64 batch = GetTensorDim(*in_backprop, data_format, 'N');
    const int64 input_rows = GetTensorDim(*in_backprop, data_format, 'H');
    const int64 input_cols = GetTensorDim(*in_backprop, data_format, 'W');

    const int64 filter_rows = filter.dim_size(0);
    const int64 filter_cols = filter.dim_size(1);
    const int64 in_depth = filter.dim_size(2);
    const int64 out_depth = filter.dim_size(3);

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding, &out_cols, &pad_cols));

    if (stride_rows == 1 && stride_cols == 1 && filter_rows == 3 &&
        filter_cols == 1) {
      // We need to change the padding from input padding to output padding for
      // the winograd matmul kernel. pad_out = filt_size - 1 - pad_in
      SYCLConv2DParams params{
          out_depth,   in_depth,     batch,       out_rows,    out_cols,
          filter_rows, filter_cols,  stride_rows, stride_cols, input_rows,
          input_cols,  2 - pad_rows, -pad_cols};
      LaunchMatmulWinograd<T, 2, 1, 3, 1, ConvType::InputBackprop>::launch(
          context, in_backprop, out_backprop, filter, params);
      return;
    }
    if (stride_rows == 1 && stride_cols == 1 && filter_rows == 1 &&
        filter_cols == 3) {
      SYCLConv2DParams params{
          out_depth,   in_depth,    batch,       out_rows,    out_cols,
          filter_rows, filter_cols, stride_rows, stride_cols, input_rows,
          input_cols,  -pad_rows,   2 - pad_cols};
      LaunchMatmulWinograd<T, 1, 2, 1, 3, ConvType::InputBackprop>::launch(
          context, in_backprop, out_backprop, filter, params);
      return;
    }
    if (stride_rows == 1 && stride_cols == 1 && filter_rows == 3 &&
        filter_cols == 3) {
      SYCLConv2DParams params{
          out_depth,   in_depth,     batch,       out_rows,    out_cols,
          filter_rows, filter_cols,  stride_rows, stride_cols, input_rows,
          input_cols,  2 - pad_rows, 2 - pad_cols};
      LaunchMatmulWinograd<T, 2, 2, 3, 3, ConvType::InputBackprop>::launch(
          context, in_backprop, out_backprop, filter, params);
      return;
    }

    SYCLConv2DParams params{in_depth,    out_depth,   batch,       input_rows,
                            input_cols,  filter_rows, filter_cols, stride_rows,
                            stride_cols, out_rows,    out_cols,    pad_rows,
                            pad_cols};

    LaunchIm2Col<T, ConvType::InputBackprop>::launch(
        context, in_backprop, out_backprop, filter, params);
    // Note: The input and output depths are switched for the naive input
    // backprop
    //SYCLConv2DParams params{out_depth,   in_depth,    batch,       input_rows,
    //                        input_cols,  filter_rows, filter_cols, stride_rows,
    //                        stride_cols, out_rows,    out_cols,    pad_rows,
    //                        pad_cols};
    //LaunchConv2DBackpropInputSYCL<T>::launch(context, in_backprop, out_backprop,
    //                                         filter, params);
  }

 private:
  /**
   * Check whether the convolution can be computed with a single matrix
   * multiply. This is the case when the filter is 1x1 or where the filter is
   * the same size as the input tensor.
   *
   * The MatMulConvFunctor used here is defined in
   * tensorflow/core/kernels/conv_2d.h and just calls Eigen contract.
   *
   * Returns true if the convolution has been launched, and false if the
   * convolution cannot be computed using a matrix multiply.
   */
  bool launch_matmul(OpKernelContext* context, const Tensor& out_backprop,
                     const Tensor& filter, int64 stride_rows, int64 stride_cols,
                     const Padding& padding, Tensor* in_backprop) {
    if (filter.dim_size(0) == 1 && filter.dim_size(1) == 1 &&
        stride_rows == 1 && stride_cols == 1) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      int conv_width = 1;
      for (int i = 0; i < 3; ++i) {
        conv_width *= in_backprop->dim_size(i);
      }

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 1);
      functor::MatMulConvFunctor<SYCLDevice, T>()(
          context->eigen_device<SYCLDevice>(),
          in_backprop->shaped<T, 2>({conv_width, filter.dim_size(2)}),
          out_backprop.shaped<T, 2>({conv_width, filter.dim_size(3)}),
          filter.shaped<T, 2>({filter.dim_size(2), filter.dim_size(3)}),
          dim_pair);
      return true;
    } else {
      return false;
    }
  }
};
template <typename T>
struct LaunchConv2DBackpropFilterOp<SYCLDevice, T> {
  void operator()(OpKernelContext* context, bool /*use_cudnn*/,
                  bool /*cudnn_use_autotune*/, const Tensor& out_backprop,
                  const Tensor& input, int64 stride_rows, int64 stride_cols,
                  const Padding& padding, Tensor* filter_backprop,
                  TensorFormat data_format) {
    if (launch_matmul(context, out_backprop, input, stride_rows, stride_cols,
                      padding, filter_backprop)) {
      return;
    }
    const int64 batch = GetTensorDim(input, data_format, 'N');
    const int64 input_rows = GetTensorDim(input, data_format, 'H');
    const int64 input_cols = GetTensorDim(input, data_format, 'W');

    const int64 filter_rows = filter_backprop->dim_size(0);
    const int64 filter_cols = filter_backprop->dim_size(1);
    const int64 in_depth = filter_backprop->dim_size(2);
    const int64 out_depth = filter_backprop->dim_size(3);

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding, &out_cols, &pad_cols));

    // Need to map the dimensions provided by TF to those we expect in the
    // convolution kernel.
    const int64 window_rows = out_rows * stride_rows - (stride_rows - 1);
    const int64 window_cols = out_cols * stride_cols - (stride_cols - 1);

    SYCLConv2DParams params{in_depth,    out_depth,   batch,       input_rows,
                            input_cols,  window_rows, window_cols, stride_rows,
                            stride_cols, filter_rows, filter_cols, pad_rows,
                            pad_cols};

    if (stride_rows == 1 && stride_cols == 1 && filter_rows == 1 &&
        filter_cols == 3) {
      LaunchMatmulWinograd<T, 1, 3, 1, 2, ConvType::FilterBackprop>::launch(
          context, filter_backprop, input, out_backprop, params);
      return;
    }
    if (stride_rows == 1 && stride_cols == 1 && filter_rows == 3 &&
        filter_cols == 1) {
      LaunchMatmulWinograd<T, 3, 1, 2, 1, ConvType::FilterBackprop>::launch(
          context, filter_backprop, input, out_backprop, params);
      return;
    }
    if (stride_rows == 1 && stride_cols == 1 && filter_rows == 3 &&
        filter_cols == 3) {
      LaunchMatmulWinograd<T, 3, 3, 2, 2, ConvType::FilterBackprop>::launch(
          context, filter_backprop, input, out_backprop, params);
      return;
    }

    LaunchConv2DBackpropFilterSYCL<T>::launch(context, filter_backprop, input,
                                              out_backprop, params);
  }

 private:
  /**
   * Check whether the convolution can be computed with a single matrix
   * multiply. This is the case when the filter is 1x1 or where the filter is
   * the same size as the input tensor.
   *
   * The MatMulConvFunctor used here is defined in
   * tensorflow/core/kernels/conv_2d.h and just calls Eigen contract.
   *
   * Returns true if the convolution has been launched, and false if the
   * convolution cannot be computed using a matrix multiply.
   */
  bool launch_matmul(OpKernelContext* context, const Tensor& out_backprop,
                     const Tensor& input, int64 stride_rows, int64 stride_cols,
                     const Padding& padding, Tensor* filter_backprop) {
    if (filter_backprop->dim_size(0) == 1 &&
        filter_backprop->dim_size(1) == 1 && stride_rows == 1 &&
        stride_cols == 1) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      int conv_width = 1;
      for (int i = 0; i < 3; ++i) {
        conv_width *= input.dim_size(i);
      }

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(0, 0);
      functor::MatMulConvFunctor<SYCLDevice, T>()(
          context->eigen_device<SYCLDevice>(),
          filter_backprop->shaped<T, 2>(
              {filter_backprop->dim_size(2), filter_backprop->dim_size(3)}),
          input.shaped<T, 2>({conv_width, filter_backprop->dim_size(2)}),
          out_backprop.shaped<T, 2>({conv_width, filter_backprop->dim_size(3)}),
          dim_pair);
      return true;
    } else {
      return false;
    }
  }
};
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_
