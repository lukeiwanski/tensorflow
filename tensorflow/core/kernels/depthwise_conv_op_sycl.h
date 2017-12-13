#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_DEPTHWISE_CONV_OP_SYCL_H_
#define TENSORFLOW_KERNELS_DEPTHWISE_CONV_OP_SYCL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_ops_sycl_common.h"
#include "tensorflow/core/kernels/depthwise_conv_op.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
namespace sycl_conv {
struct DepthwiseConv2DParams {
  using Index = int;
  //  DepthwiseConv2DParams() = default;
  //  DepthwiseConv2DParams(DepthwiseConv2DParams const&) = default;
  //  DepthwiseConv2DParams(DepthwiseConv2DParams&&) = default;

  Index batch_;

  Index in_rows_;
  Index in_cols_;
  Index channels_;
  Index channel_multiplier_;

  Index window_rows_;
  Index window_cols_;

  Index stride_rows_;
  Index stride_cols_;

  Index out_rows_;
  Index out_cols_;
  Index out_depth_;

  Index pad_rows_;
  Index pad_cols_;

  /**
   * Get the index in the kernel tensor for a particular channel, row and
   * column.
   */
  inline TF_ATTRIBUTE_ALWAYS_INLINE Index kernel_index(const Index channel,
                                                       const Index multiplier,
                                                       const Index i,
                                                       const Index j) const {
    return (((i * window_cols_) + j) * channels_ + channel) *
               channel_multiplier_ +
           multiplier;
  }
  /**
   * Get the index in the kernel tensor for the kernel backprop for a
   * particular channel, row and column. Here we have to mirror the kernel
   * indices to match how the backprop is computed.
   */
  inline TF_ATTRIBUTE_ALWAYS_INLINE Index backprop_index(const Index channel,
                                                         const Index multiplier,
                                                         const Index i,
                                                         const Index j) const {
    const Index mirrored_row = window_rows_ - i - 1;
    const Index mirrored_col = window_cols_ - j - 1;
    return ((mirrored_row * window_cols_ + mirrored_col) * channels_ +
            channel) *
               channel_multiplier_ +
           multiplier;
  }
  /**
   * For the filter backprop we are using the output tensor as the filter of
   * the convolution, which has dimensions NHWC, rather than the filter
   * dimensions HWCF, so the kernel index is computed in a different way.
   */
  inline TF_ATTRIBUTE_ALWAYS_INLINE Index
  filter_kernel_index(const Index batch, const Index i, const Index j,
                      const Index feature) const {
    const Index filter_rows = RoundRatioUpAboveZero(window_rows_, stride_rows_);
    const Index filter_cols = RoundRatioUpAboveZero(window_cols_, stride_cols_);
    return ((batch * filter_rows + i) * filter_cols + j) * out_depth_ + feature;
  }
  /**
   * Get the window in the input tensor which corresponds to the specified
   * output index.
   *
   * NOTE: The index types used here must be signed to ensure that the padding
   * is correctly calculated.
   */
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCL2DWindow
  input_window_from_output(const Index tile_idx) const {
    static_assert(std::is_integral<Index>::value,
                  "Index must be an integral type");
    static_assert(std::is_signed<Index>::value, "Index must be a signed type");
    Index batch = tile_idx;
    const Index cstart = (batch % out_cols_) * stride_cols_ - pad_cols_;
    const Index cend = cl::sycl::min(cstart + window_cols_, in_cols_);
    const Index firstc = cstart < 0 ? -cstart : 0;
    batch /= out_cols_;

    const Index rstart = (batch % out_rows_) * stride_rows_ - pad_rows_;
    const Index rend = cl::sycl::min(rstart + window_rows_, in_rows_);
    const Index firstr = rstart < 0 ? -rstart : 0;
    batch /= out_rows_;

    return {rstart, rend, firstr, cstart, cend, firstc, batch};
  }
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCL2DWindow
  output_window_from_input_no_dilation(const Index index) const {
    Index n = index;
    // c is the index in the padded output tensor (ie with lots of extra zeros),
    // but without the first padding. first_padded_c adds this extra padding.
    const Index c = (n % in_cols_) + pad_cols_;
    const Index first_padded_c = c - window_cols_ + 1;
    // The first and last output indices affected by this input.
    const Index last_used_c = c / stride_cols_;
    const Index first_used_c =
        RoundRatioUpAboveZero(first_padded_c, stride_cols_);

    const Index offset_c = first_used_c * stride_cols_ - first_padded_c;
    const Index cstart = cl::sycl::max(first_used_c, static_cast<Index>(0));
    const Index cend = cl::sycl::min(last_used_c + 1, out_cols_);
    n /= in_cols_;

    const Index r = (n % in_rows_) + pad_rows_;
    const Index last_used_r = r / stride_rows_;
    const Index first_padded_r = r - window_rows_ + 1;
    const Index first_used_r =
        RoundRatioUpAboveZero(first_padded_r, stride_rows_);

    const Index offset_r = first_used_r * stride_rows_ - first_padded_r;
    const Index rstart = cl::sycl::max(first_used_r, static_cast<Index>(0));
    const Index rend = cl::sycl::min(last_used_r + 1, out_rows_);
    n /= in_rows_;

    return {rstart, rend, offset_r, cstart, cend, offset_c, n};
  }
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCL2DKernelWindow
  kernel_window_from_output(const Index index) const {
    static_assert(std::is_integral<Index>::value,
                  "Index must be an integral type");
    static_assert(std::is_signed<Index>::value, "Index must be a signed type");
    Index n = index;

    Index cstart = n % out_cols_ - pad_cols_;
    const Index cend = cl::sycl::min(cstart + window_cols_, in_cols_);
    const Index firstc = cstart < 0 ? -cstart : 0;
    n /= out_cols_;

    Index rstart = n - pad_rows_;
    const Index rend = cl::sycl::min(rstart + window_rows_, in_rows_);
    const Index firstr = rstart < 0 ? -rstart : 0;

    return {rstart, rend, firstr, cstart, cend, firstc, 0, 0};
  }
};
template <typename T, ConvType CType>
struct DepthwiseConv2D;
template <typename T>
struct DepthwiseConv2D<T, ConvType::Forward> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE DepthwiseConv2D(
      Index n_elems, const DepthwiseConv2DParams& params,
      const read_accessor input, const read_accessor kernel,
      write_accessor output)
      : n_elems_{n_elems},
        p_(params),
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const Index index = item.get(0);

    if (index < n_elems_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index out_channel = index % p_.out_depth_;
      const Index tile_idx = index / p_.out_depth_;
      const Index multiple = out_channel % p_.channel_multiplier_;
      const Index channel = out_channel / p_.channel_multiplier_;
      const SYCL2DWindow w = p_.input_window_from_output(tile_idx);

      T out_val = static_cast<T>(0);
      const T* input_data_n =
          input_data + w.batch * p_.in_cols_ * p_.in_rows_ * p_.channels_;
      for (Index r = w.rstart, i = 0; r < w.rend; ++r, ++i) {
        if (r >= 0) {
          for (Index c = w.cstart, j = 0; c < w.cend; ++c, ++j) {
            if (c >= 0) {
              const Index idx = (r * p_.in_cols_ + c) * p_.channels_ + channel;
              const Index k_idx = p_.kernel_index(channel, multiple, i, j);
              out_val += input_data_n[idx] * kernel_data[k_idx];
            }
          }
        }
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const DepthwiseConv2DParams p_;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
template <typename T>
struct DepthwiseConv2D<T, ConvType::InputBackprop> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE DepthwiseConv2D(
      Index n_elems, const DepthwiseConv2DParams& params,
      const read_accessor input, const read_accessor kernel,
      write_accessor output)
      : n_elems_{n_elems},
        p_(params),
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const Index index = item.get(0);
    if (index < n_elems_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index channel = index % p_.channels_;
      const Index tile_idx = index / p_.channels_;
      const SYCL2DWindow w = p_.output_window_from_input_no_dilation(tile_idx);

      T out_val = static_cast<T>(0);
      const T* input_data_n =
          input_data + w.batch * p_.out_cols_ * p_.out_rows_ * p_.out_depth_;
      for (Index r = w.rstart, i = w.firstr; r < w.rend;
           ++r, i += p_.stride_rows_) {
        for (Index c = w.cstart, j = w.firstc; c < w.cend;
             ++c, j += p_.stride_cols_) {
          for (Index multiple = 0; multiple < p_.channel_multiplier_;
               ++multiple) {
            const Index idx =
                ((r * p_.out_cols_ + c) * p_.channels_ + channel) *
                    p_.channel_multiplier_ +
                multiple;
            const Index k_idx = p_.backprop_index(channel, multiple, i, j);
            out_val += input_data_n[idx] * kernel_data[k_idx];
          }
        }
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const DepthwiseConv2DParams p_;
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
struct DepthwiseConv2D<T, ConvType::FilterBackprop> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE DepthwiseConv2D(
      Index n_elems, const DepthwiseConv2DParams& params,
      const read_accessor input, const read_accessor kernel,
      write_accessor output)
      : n_elems_{n_elems},
        p_(params),
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const Index index = item.get(0);
    if (index < n_elems_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      Index t_idx = index;
      const Index multiple = t_idx % p_.channel_multiplier_;
      t_idx /= p_.channel_multiplier_;
      const Index channel = t_idx % p_.channels_;
      t_idx /= p_.channels_;
      const SYCL2DKernelWindow w = p_.kernel_window_from_output(t_idx);

      T out_val = static_cast<T>(0);
      const T* input_data_n = input_data;
      for (Index b = 0; b < p_.batch_; b++) {
        for (Index r = w.rstart, i = 0; r < w.rend; ++i, r += p_.stride_rows_) {
          if (r >= 0) {
            for (Index c = w.cstart, j = 0; c < w.cend;
                 ++j, c += p_.stride_cols_) {
              if (c >= 0) {
                const Index idx =
                    (r * p_.in_cols_ + c) * p_.channels_ + channel;
                const Index k_idx = p_.filter_kernel_index(
                    b, i, j, channel * p_.channel_multiplier_ + multiple);
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
  const DepthwiseConv2DParams p_;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
template <ConvType CType>
inline DepthwiseConv2DParams get_kernel_params(DepthwiseConv2DParams params) {
  return params;
}
template <>
inline DepthwiseConv2DParams get_kernel_params<ConvType::FilterBackprop>(
    DepthwiseConv2DParams params) {
  // Map the input dimensions to those expected in the convolution kernel.
  const auto window_rows =
      params.out_rows_ * params.stride_rows_ - (params.stride_rows_ - 1);
  const auto window_cols =
      params.out_cols_ * params.stride_cols_ - (params.stride_cols_ - 1);
  params.out_rows_ = params.window_rows_;
  params.out_cols_ = params.window_cols_;
  params.window_rows_ = window_rows;
  params.window_cols_ = window_cols;
  return params;
}
template <ConvType CType>
inline size_t get_output_size(DepthwiseConv2DParams const& params);
template <>
inline size_t get_output_size<ConvType::Forward>(
    DepthwiseConv2DParams const& params) {
  return params.batch_ * params.out_rows_ * params.out_cols_ *
         params.out_depth_;
}
template <>
inline size_t get_output_size<ConvType::InputBackprop>(
    DepthwiseConv2DParams const& params) {
  return params.batch_ * params.in_rows_ * params.in_cols_ * params.channels_;
}
template <>
inline size_t get_output_size<ConvType::FilterBackprop>(
    DepthwiseConv2DParams const& params) {
  return params.window_rows_ * params.window_cols_ * params.out_depth_;
}
template <typename T, ConvType CType>
struct LaunchDepthwiseConv2DKernel {
  using Functor = DepthwiseConv2D<T, CType>;
  static constexpr auto read_mode = Functor::read_mode;
  static constexpr auto write_mode = Functor::write_mode;
  using Index = int;

  static void launch(Eigen::SyclDevice const& device, T* const output,
                     T const* const input, T const* const filter,
                     DepthwiseConv2DParams const& params) {
    const Index output_size = get_output_size<CType>(params);
    const Index workgroup_size = device.maxSyclThreadsPerBlock();
    const Index n_threads =
        RoundUpToNearestMultiple(output_size, workgroup_size);

    auto input_buffer = device.get_sycl_buffer(input);
    auto filter_buffer = device.get_sycl_buffer(filter);
    auto output_buffer = device.get_sycl_buffer(output);
    DepthwiseConv2DParams kernel_params = get_kernel_params<CType>(params);

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access = input_buffer.template get_access<read_mode>(cgh);
      auto filter_access = filter_buffer.template get_access<read_mode>(cgh);
      auto output_access = output_buffer.template get_access<write_mode>(cgh);

      Functor conv(output_size, kernel_params, input_access, filter_access,
                   output_access);

      cgh.parallel_for(cl::sycl::range<1>(n_threads), conv);
    });
  }
};
template <typename T, ConvType CType>
struct DLauncher final : public LaunchDepthwiseConv2DKernel<T, CType> {};
}  // namespace sycl_conv
template <typename T>
struct LaunchDepthwiseConvOp<SYCLDevice, T> {
  void operator()(OpKernelContext* ctx, const DepthwiseArgs& args,
                  const T* input, const T* depthwise_filter, T* output,
                  TensorFormat data_format) {
    OP_REQUIRES(
        ctx, data_format == FORMAT_NHWC,
        errors::Unimplemented(
            "Depthwise convolution on SYCL is only supported for NHWC format"));
    sycl_conv::DepthwiseConv2DParams params;
    params.batch_ = args.batch;
    params.in_rows_ = args.in_rows;
    params.in_cols_ = args.in_cols;
    params.channels_ = args.in_depth;
    params.window_rows_ = args.filter_rows;
    params.window_cols_ = args.filter_cols;
    params.channel_multiplier_ = args.depth_multiplier;
    params.stride_rows_ = args.stride;
    params.stride_cols_ = args.stride;
    params.pad_rows_ = args.pad_rows;
    params.pad_cols_ = args.pad_cols;
    params.out_rows_ = args.out_rows;
    params.out_cols_ = args.out_cols;
    params.out_depth_ = args.out_depth;
    sycl_conv::DLauncher<T, ConvType::Forward>::launch(
        ctx->eigen_device<SYCLDevice>(), output, input, depthwise_filter,
        params);
  }
};
template <typename T>
struct LaunchDepthwiseConvBackpropInputOp<SYCLDevice, T> {
  void operator()(OpKernelContext* ctx, const DepthwiseArgs& args,
                  const T* out_backprop, const T* depthwise_filter,
                  T* in_backprop, TensorFormat data_format) {
    OP_REQUIRES(
        ctx, data_format == FORMAT_NHWC,
        errors::Unimplemented(
            "Depthwise convolution on SYCL is only supported for NHWC format"));
    sycl_conv::DepthwiseConv2DParams params;
    params.batch_ = args.batch;
    params.in_rows_ = args.in_rows;
    params.in_cols_ = args.in_cols;
    params.channels_ = args.in_depth;
    params.window_rows_ = args.filter_rows;
    params.window_cols_ = args.filter_cols;
    params.channel_multiplier_ = args.depth_multiplier;
    params.stride_rows_ = args.stride;
    params.stride_cols_ = args.stride;
    params.pad_rows_ = args.pad_rows;
    params.pad_cols_ = args.pad_cols;
    params.out_rows_ = args.out_rows;
    params.out_cols_ = args.out_cols;
    params.out_depth_ = args.out_depth;
    sycl_conv::DLauncher<T, ConvType::InputBackprop>::launch(
        ctx->eigen_device<SYCLDevice>(), in_backprop, out_backprop,
        depthwise_filter, params);
  }
};
template <typename T>
struct LaunchDepthwiseConvBackpropFilterOp<SYCLDevice, T> {
  void operator()(OpKernelContext* ctx, const DepthwiseArgs& args,
                  const T* out_backprop, const T* input, T* filter_backprop,
                  TensorFormat data_format) {
    OP_REQUIRES(
        ctx, data_format == FORMAT_NHWC,
        errors::Unimplemented(
            "Depthwise convolution on SYCL is only supported for NHWC format"));
    sycl_conv::DepthwiseConv2DParams params;
    params.batch_ = args.batch;
    params.in_rows_ = args.in_rows;
    params.in_cols_ = args.in_cols;
    params.channels_ = args.in_depth;
    params.window_rows_ = args.filter_rows;
    params.window_cols_ = args.filter_cols;
    params.channel_multiplier_ = args.depth_multiplier;
    params.stride_rows_ = args.stride;
    params.stride_cols_ = args.stride;
    params.pad_rows_ = args.pad_rows;
    params.pad_cols_ = args.pad_cols;
    params.out_rows_ = args.out_rows;
    params.out_cols_ = args.out_cols;
    params.out_depth_ = args.out_depth;
    sycl_conv::DLauncher<T, ConvType::FilterBackprop>::launch(
        ctx->eigen_device<SYCLDevice>(), filter_backprop, input, out_backprop,
        params);
  }
};
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_DEPTHWISE_CONV_OP_SYCL_H_
