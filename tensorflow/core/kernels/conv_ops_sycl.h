/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_grad_ops.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
// Forward declarations needed for later specializations.
template <typename Device, typename T>
struct LaunchConv2DOp;
/**
 * Helper function to provide the ratio of two integers, always rounded up. If
 * the numerator is negative then we assume that the rounded ratio will be
 * zero, otherwise we need to ensure that the value is rounded up rather
 * than down.
 */
template <typename IntegerType>
inline TF_ATTRIBUTE_ALWAYS_INLINE IntegerType
RoundRatioUpAboveZero(const IntegerType num, const IntegerType div) {
  return num < 0 ? 0 : (num + div - 1) / div;
}
template <typename IntegerType>
inline TF_ATTRIBUTE_ALWAYS_INLINE IntegerType
RoundUpToNearestMultiple(const IntegerType val, const IntegerType multiplier) {
  const IntegerType diff = val % multiplier;
  return val + (multiplier - diff);
}

struct SYCL2DWindow {
  using Index = int;

  const Index rstart;
  const Index rend;
  const Index firstr;

  const Index cstart;
  const Index cend;
  const Index firstc;

  const Index feature;
  const Index batch;
};
struct SYCL2DKernelWindow {
  using Index = int;

  const Index rstart;
  const Index rend;
  const Index firstr;

  const Index cstart;
  const Index cend;
  const Index firstc;

  const Index feature;
  const Index channel;
};
struct SYCLConv2DParams {
  using Index = int;

  template <typename _Index>
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCLConv2DParams(
      const _Index channels, const _Index features, const _Index batch,
      const _Index in_rows, const _Index in_cols, const _Index window_rows,
      const _Index window_cols, const _Index stride_rows,
      const _Index stride_cols, const _Index out_rows, const _Index out_cols,
      const _Index pad_rows, const _Index pad_cols)
      :

        channels_{static_cast<Index>(channels)},
        features_{static_cast<Index>(features)},
        batch_{static_cast<Index>(batch)},
        in_rows_{static_cast<Index>(in_rows)},
        in_cols_{static_cast<Index>(in_cols)},
        window_rows_{static_cast<Index>(window_rows)},
        window_cols_{static_cast<Index>(window_cols)},
        stride_rows_{static_cast<Index>(stride_rows)},
        stride_cols_{static_cast<Index>(stride_cols)},
        out_rows_{static_cast<Index>(out_rows)},
        out_cols_{static_cast<Index>(out_cols)},
        pad_rows_{static_cast<Index>(pad_rows)},
        pad_cols_{static_cast<Index>(pad_cols)} {}

  /* The number of input channels. */
  const Index channels_;
  /* The number of output feature channels. */
  const Index features_;
  const Index batch_;

  const Index in_rows_;
  const Index in_cols_;

  const Index window_rows_;
  const Index window_cols_;

  const Index stride_rows_;
  const Index stride_cols_;

  const Index out_rows_;
  const Index out_cols_;

  const Index pad_rows_;
  const Index pad_cols_;

  /**
   * Get the index in the kernel tensor for a particular channel, row and
   * column.
   */
  inline TF_ATTRIBUTE_ALWAYS_INLINE Index kernel_index(const Index channel,
                                                       const Index feature,
                                                       const Index i,
                                                       const Index j) const {
    return (((i * window_cols_) + j) * channels_ + channel) * features_ +
           feature;
  }
  /**
   * Get the index in the kernel tensor for the kernel backprop for a
   * particular channel, row and column. Here we have to mirror the kernel
   * indices to match how the backprop is ocmputed.
   */
  inline TF_ATTRIBUTE_ALWAYS_INLINE Index backprop_index(const Index channel,
                                                         const Index feature,
                                                         const Index i,
                                                         const Index j) const {
    const Index mirrored_row = window_rows_ - i - 1;
    const Index mirrored_col = window_cols_ - j - 1;
    return ((mirrored_row * window_cols_ + mirrored_col) * channels_ +
            channel) *
               features_ +
           feature;
  }
  /**
   * For the filter backprop we are using the output tensor as the filter of
   * the convolution, which has dimesnsions NHWC, rather than the fiter
   * dimensions HWCF, so the kernel index is computed in a different way.
   */
  inline TF_ATTRIBUTE_ALWAYS_INLINE Index
  filter_kernel_index(const Index batch, const Index i, const Index j,
                      const Index feature) const {
    const Index filter_rows = RoundRatioUpAboveZero(window_rows_, stride_rows_);
    const Index filter_cols = RoundRatioUpAboveZero(window_cols_, stride_cols_);
    return ((batch * filter_rows + i) * filter_cols + j) * features_ + feature;
  }
  /**
   * Get the window in the input tensor which corresponds to the specified
   * output index.
   *
   * NOTE: The index types used here must be signed to ensure that the padding
   * is correctly calculated.
   */
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCL2DWindow
  input_window_from_output(const Index index) const {
    Index batch = index;
    const Index feature = batch % features_;
    batch /= features_;

    Index cstart = (batch % out_cols_) * stride_cols_ - pad_cols_;
    const Index cend = std::min(cstart + window_cols_, in_cols_);
    const Index firstc = cstart < 0 ? -cstart : 0;
    cstart = std::max(cstart, static_cast<Index>(0));
    batch /= out_cols_;

    Index rstart = (batch % out_rows_) * stride_rows_ - pad_rows_;
    const Index rend = std::min(rstart + window_rows_, in_rows_);
    const Index firstr = rstart < 0 ? -rstart : 0;
    rstart = std::max(rstart, static_cast<Index>(0));
    batch /= out_rows_;

    return {rstart, rend, firstr, cstart, cend, firstc, feature, batch};
  }
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCL2DWindow
  output_window_from_input(const Index index) const {
    Index n = index;
    const Index d = n % channels_;
    n /= channels_;

    // c is the index in the padded output tensor (ie with lots of extra zeros),
    // but without the first padding. first_padded_c adds this extra padding.
    const Index c = (n % in_cols_) + pad_cols_;
    const Index first_padded_c = c - window_cols_ + 1;
    // The first and last output indices affected by this input.
    const Index last_used_c = c / stride_cols_;
    const Index first_used_c =
        RoundRatioUpAboveZero(first_padded_c, stride_cols_);

    const Index offset_c = first_used_c * stride_cols_ - first_padded_c;
    const Index cstart = std::max(first_used_c, static_cast<Index>(0));
    const Index cend = std::min(last_used_c + 1, out_cols_);
    n /= in_cols_;

    const Index r = (n % in_rows_) + pad_rows_;
    const Index last_used_r = r / stride_rows_;
    const Index first_padded_r = r - window_rows_ + 1;
    const Index first_used_r =
        RoundRatioUpAboveZero(first_padded_r, stride_rows_);

    const Index offset_r = first_used_r * stride_rows_ - first_padded_r;
    const Index rstart = std::max(first_used_r, static_cast<Index>(0));
    const Index rend = std::min(last_used_r + 1, out_rows_);
    n /= in_rows_;

    return {rstart, rend, offset_r, cstart, cend, offset_c, d, n};
  }
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCL2DKernelWindow
  kernel_window_from_output(const Index index) const {
    Index n = index;
    const Index feature = n % features_;
    n /= features_;
    const Index channel = n % channels_;
    n /= channels_;

    Index cstart = n % out_cols_ - pad_cols_;
    const Index cend = std::min(cstart + window_cols_, in_cols_);
    const Index firstc = cstart < 0 ? -cstart : 0;
    while (cstart < 0) {
      cstart += stride_cols_;
    }
    n /= out_cols_;

    Index rstart = n - pad_rows_;
    const Index rend = std::min(rstart + window_rows_, in_rows_);
    const Index firstr = rstart < 0 ? -rstart : 0;
    while (rstart < 0) {
      rstart += stride_rows_;
    }

    return {rstart, rend, firstr, cstart, cend, firstc, feature, channel};
  }
};
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
      SYCL2DWindow w = p_.input_window_from_output(index);

      T out_val = static_cast<T>(0);
      const T* input_data_n =
          input_data + w.batch * p_.in_cols_ * p_.in_rows_ * p_.channels_;
      for (Index r = w.rstart, i = w.firstr; r < w.rend; ++r, ++i) {
        for (Index c = w.cstart, j = w.firstc; c < w.cend; ++c, ++j) {
          for (Index channel = 0; channel < p_.channels_; ++channel) {
            const Index idx = (r * p_.in_cols_ + c) * p_.channels_ + channel;
            const Index k_idx = p_.kernel_index(channel, w.feature, i, j);
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
      SYCL2DWindow w = p_.output_window_from_input(index);

      T out_val = static_cast<T>(0);
      const T* input_data_n =
          input_data + w.batch * p_.out_cols_ * p_.out_rows_ * p_.features_;
      for (Index r = w.rstart, i = w.firstr; r < w.rend;
           ++r, i += p_.stride_rows_) {
        for (Index c = w.cstart, j = w.firstc; c < w.cend;
             ++c, j += p_.stride_cols_) {
          for (Index feature = 0; feature < p_.features_; ++feature) {
            const Index idx = (r * p_.out_cols_ + c) * p_.features_ + feature;
            const Index k_idx = p_.backprop_index(w.feature, feature, i, j);
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
      SYCL2DKernelWindow w = p_.kernel_window_from_output(index);

      T out_val = static_cast<T>(0);
      const T* input_data_n = input_data;
      for (Index b = 0; b < p_.batch_; b++) {
        for (Index r = w.rstart, i = w.firstr; r < w.rend;
             ++i, r += p_.stride_rows_) {
          for (Index c = w.cstart, j = w.firstc; c < w.cend;
               ++j, c += p_.stride_cols_) {
            const Index idx = (r * p_.in_cols_ + c) * p_.channels_ + w.channel;
            const Index k_idx = p_.filter_kernel_index(b, i, j, w.feature);
            out_val += input_data_n[idx] * kernel_data[k_idx];
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
struct LaunchConv2DSYCL : public LaunchConv2DKernel<T, functor::Conv2DSYCL<T>> {
};

template <typename T>
struct LaunchConv2DBackpropInputSYCL
    : public LaunchConv2DKernel<T, functor::Conv2DBackpropSYCL<T>> {};

template <typename T>
struct LaunchConv2DBackpropFilterSYCL
    : public LaunchConv2DKernel<T, functor::Conv2DBackpropFilterSYCL<T>> {};

template <typename T>
struct LaunchConv2DOp<SYCLDevice, T> {
  void operator()(OpKernelContext* context, bool use_cudnn,
                  bool cudnn_use_autotune, const Tensor& input,
                  const Tensor& filter, int64 stride_rows, int64 stride_cols,
                  const Padding& padding, Tensor* output,
                  TensorFormat data_format) {
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

    LaunchConv2DSYCL<T>::launch(context, output, input, filter, params);
  }
};
template <typename T>
struct LaunchConv2DBackpropInputOp<SYCLDevice, T> {
  void operator()(OpKernelContext* context, bool /*use_cudnn*/,
                  bool /*cudnn_use_autotune*/, const Tensor& out_backprop,
                  const Tensor& filter, int64 stride_rows, int64 stride_cols,
                  const Padding& padding, Tensor* in_backprop,
                  TensorFormat data_format) {
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

    SYCLConv2DParams params{in_depth,    out_depth,   batch,       input_rows,
                            input_cols,  filter_rows, filter_cols, stride_rows,
                            stride_cols, out_rows,    out_cols,    pad_rows,
                            pad_cols};

    LaunchConv2DBackpropInputSYCL<T>::launch(context, in_backprop, out_backprop,
                                             filter, params);
  }
};
template <typename T>
struct LaunchConv2DBackpropFilterOp<SYCLDevice, T> {
  void operator()(OpKernelContext* context, bool /*use_cudnn*/,
                  bool /*cudnn_use_autotune*/, const Tensor& out_backprop,
                  const Tensor& input, int64 stride_rows, int64 stride_cols,
                  const Padding& padding, Tensor* filter_backprop,
                  TensorFormat data_format) {
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

    LaunchConv2DBackpropFilterSYCL<T>::launch(context, filter_backprop, input,
                                              out_backprop, params);
  }
};
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_
