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

#include "tensorflow/core/kernels/conv_ops_sycl_common.h"
#include "tensorflow/core/kernels/conv_ops_winograd_sycl.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
// Forward declarations needed for later specializations.
template <typename Device, typename T>
struct LaunchConv2DOp;
namespace functor {
/**
 * SYCL kernel for naive convolution computation.
 */
template <typename T>
struct Conv2DSYCL {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE Conv2DSYCL(Index n_elems,
                                               const SYCLConv2DParams& params,
                                               const read_accessor input,
                                               const read_accessor kernel,
                                               write_accessor output)
      : n_elems_{n_elems},
        p_{params},
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
    const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_accessor_);
    const Index index = item.get(0);

    if (index < n_elems_) {
      const SYCL2DWindow w = p_.input_window_from_output(index);

      T out_val = static_cast<T>(0);
      const T* input_data_n =
          input_data + w.batch * p_.in_cols_ * p_.in_rows_ * p_.channels_;
      for (Index r = w.rstart, i = 0; r < w.rend; ++r, ++i) {
        if (r >= 0) {
          for (Index c = w.cstart, j = 0; c < w.cend; ++c, ++j) {
            if (c >= 0) {
              for (Index channel = 0; channel < p_.channels_; ++channel) {
                const Index idx =
                    (r * p_.in_cols_ + c) * p_.channels_ + channel;
                const Index k_idx = p_.kernel_index(channel, w.feature, i, j);
                out_val += input_data_n[idx] * kernel_data[k_idx];
              }
            }
          }
        }
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
template <typename T>
struct Conv2DBackpropSYCL {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE Conv2DBackpropSYCL(
      Index n_elems, const SYCLConv2DParams& params, const read_accessor input,
      const read_accessor kernel, write_accessor output)
      : n_elems_{n_elems},
        p_{params},
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
    const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_accessor_);
    const Index index = item.get(0);

    if (index < n_elems_) {
      const SYCL2DWindow w = p_.output_window_from_input(index);

      T out_val = static_cast<T>(0);
      const T* input_data_n =
          input_data + w.batch * p_.in_cols_ * p_.in_rows_ * p_.channels_;
      for (Index r = w.rstart, i = w.firstr; r < w.rend;
           ++r, i += p_.stride_rows_) {
        for (Index c = w.cstart, j = w.firstc; c < w.cend;
             ++c, j += p_.stride_cols_) {
          for (Index channel = 0; channel < p_.channels_; ++channel) {
            const Index idx = (r * p_.in_cols_ + c) * p_.channels_ + channel;
            const Index k_idx = p_.backprop_index(w.feature, channel, i, j);
            out_val += input_data_n[idx] * kernel_data[k_idx];
          }
        }
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
/*
 * The main difference between the two backprop kernels is the way strides are
 * handled. In the filter backprop the input is strided and the kernel is not
 * whereas in the input backprop this is the other way around.
 */
template <typename T>
struct Conv2DBackpropFilterSYCL {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE Conv2DBackpropFilterSYCL(
      Index n_elems, const SYCLConv2DParams& params, const read_accessor input,
      const read_accessor kernel, write_accessor output)
      : n_elems_{n_elems},
        p_{params},
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
    const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_accessor_);
    const Index index = item.get(0);

    if (index < n_elems_) {
      const SYCL2DKernelWindow w = p_.kernel_window_from_output(index);

      T out_val = static_cast<T>(0);
      const T* input_data_n = input_data;
      for (Index b = 0; b < p_.batch_; b++) {
        for (Index r = w.rstart, i = 0; r < w.rend; ++i, r += p_.stride_rows_) {
          if (r >= 0) {
            for (Index c = w.cstart, j = 0; c < w.cend;
                 ++j, c += p_.stride_cols_) {
              if (c >= 0) {
                const Index idx =
                    (r * p_.in_cols_ + c) * p_.channels_ + w.channel;
                const Index k_idx = p_.filter_kernel_index(b, i, j, w.feature);
                out_val += input_data_n[idx] * kernel_data[k_idx];
              }
            }
          }
        }
        input_data_n += p_.in_cols_ * p_.in_rows_ * p_.channels_;
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
}  // namespace functor

template <typename T, typename Functor>
struct LaunchConv2DKernel {
  static constexpr auto read_mode = Functor::read_mode;
  static constexpr auto write_mode = Functor::write_mode;
  using Index = int;

  static void launch(OpKernelContext* context, Tensor* output,
                     const Tensor& tensor_in, const Tensor& filter,
                     const SYCLConv2DParams& params) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    const Index output_size = output->NumElements();
    const Index workgroup_size = device.maxSyclThreadsPerBlock();
    const Index n_threads =
        RoundUpToNearestMultiple(output_size, workgroup_size);

    auto input_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto filter_buffer =
        device.get_sycl_buffer(filter.template flat<T>().data());
    auto output_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access = input_buffer.template get_access<read_mode>(cgh);
      auto filter_access = filter_buffer.template get_access<read_mode>(cgh);
      auto output_access = output_buffer.template get_access<write_mode>(cgh);

      Functor conv(output_size, params, input_access, filter_access,
                   output_access);

      cgh.parallel_for(cl::sycl::range<1>(n_threads), conv);
    });
  }
};
template <typename T>
struct LaunchConv2DSYCL final
    : public LaunchConv2DKernel<T, functor::Conv2DSYCL<T>> {};

template <typename T>
struct LaunchConv2DBackpropInputSYCL final
    : public LaunchConv2DKernel<T, functor::Conv2DBackpropSYCL<T>> {};

template <typename T>
struct LaunchConv2DBackpropFilterSYCL final
    : public LaunchConv2DKernel<T, functor::Conv2DBackpropFilterSYCL<T>> {};

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

    LaunchConv2DSYCL<T>::launch(context, output, input, filter, params);
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

    SYCLConv2DParams params{out_depth,   in_depth,    batch,       out_rows,
                            out_cols,    filter_rows, filter_cols, stride_rows,
                            stride_cols, input_rows,  input_cols,  pad_rows,
                            pad_cols};

    LaunchConv2DBackpropInputSYCL<T>::launch(context, in_backprop, out_backprop,
                                             filter, params);
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
