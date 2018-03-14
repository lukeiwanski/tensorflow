#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_DIRECT_TILED_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_DIRECT_TILED_SYCL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_ops_sycl_common.h"

#include "tensorflow/core/kernels/conv_ops_direct_tiled_sycl_kernels.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
namespace direct_tiled {
template <ConvType CType>
inline SYCLConv2DParams get_kernel_params(SYCLConv2DParams params) {
  return params;
}
template <>
inline SYCLConv2DParams get_kernel_params<ConvType::InputBackprop>(
    SYCLConv2DParams params) {
  // We need to change the padding from input padding to output padding for
  // the winograd matmul kernel. pad_out = filt_size - 1 - pad_in
  params.pad_rows_ = params.window_rows_ - 1 - params.pad_rows_;
  params.pad_cols_ = params.window_cols_ - 1 - params.pad_cols_;
  return params;
}
template <>
inline SYCLConv2DParams get_kernel_params<ConvType::FilterBackprop>(
    SYCLConv2DParams params) {
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
template <ConvType CType, int tile_rows, int tile_cols,
          int channel_vector_width, int feature_vector_width>
struct tiled_output_size {
  static size_t get(SYCLConv2DParams const& params) { return 0; }
};
template <int tile_rows, int tile_cols, int channel_vector_width,
          int feature_vector_width>
struct tiled_output_size<ConvType::Forward, tile_rows, tile_cols,
                         channel_vector_width, feature_vector_width> {
  static size_t get(SYCLConv2DParams const& params) {
    return params.batch_ * RoundRatioUpAboveZero(params.out_rows_, tile_rows) *
           RoundRatioUpAboveZero(params.out_cols_, tile_cols) *
           params.features_ / feature_vector_width;
  }
};
template <int tile_rows, int tile_cols, int channel_vector_width,
          int feature_vector_width>
struct tiled_output_size<ConvType::InputBackprop, tile_rows, tile_cols,
                         channel_vector_width, feature_vector_width> {
  static size_t get(SYCLConv2DParams const& params) {
    return params.batch_ * RoundRatioUpAboveZero(params.in_rows_, tile_rows) *
           RoundRatioUpAboveZero(params.in_cols_, tile_cols) *
           params.channels_ / channel_vector_width;
  }
};
template <ConvType CType>
inline bool no_fast_div(SYCLConv2DParams const& params, int tile_rows,
                        int tile_cols);
template <>
inline bool no_fast_div<ConvType::Forward>(SYCLConv2DParams const& params,
                                           int tile_rows, int tile_cols) {
  return params.features_ == 1 ||
         RoundRatioUpAboveZero(params.out_rows_, tile_rows) == 1 ||
         RoundRatioUpAboveZero(params.out_cols_, tile_cols) == 1;
}
template <>
inline bool no_fast_div<ConvType::InputBackprop>(SYCLConv2DParams const& params,
                                                 int tile_rows, int tile_cols) {
  return params.channels_ == 1 ||
         RoundRatioUpAboveZero(params.in_rows_, tile_rows) == 1 ||
         RoundRatioUpAboveZero(params.in_cols_, tile_cols) == 1;
}
template <>
inline bool no_fast_div<ConvType::FilterBackprop>(
    SYCLConv2DParams const& params, int tile_rows, int tile_cols) {
  return params.features_ == 1 || params.channels_ == 1 ||
         params.out_cols_ == 1;
}
template <ConvType>
inline bool use_static_conv(SYCLConv2DParams const& params, int const window,
                            int const stride, int const channel_vector,
                            int const feature_vector);
template <>
inline bool use_static_conv<ConvType::Forward>(SYCLConv2DParams const& params,
                                               int const window,
                                               int const stride,
                                               int const channel_vector,
                                               int const feature_vector) {
  return (params.window_cols_ == window && params.window_rows_ == window &&
          params.stride_rows_ == stride && params.stride_cols_ == stride &&
          params.features_ % feature_vector == 0);
}
template <>
inline bool use_static_conv<ConvType::InputBackprop>(
    SYCLConv2DParams const& params, int const window, int const stride,
    int const channel_vector, int const feature_vector) {
  return (params.window_cols_ == window && params.window_rows_ == window &&
          params.stride_rows_ == stride && params.stride_cols_ == stride &&
          params.channels_ % channel_vector == 0);
}
template <>
inline bool use_static_conv<ConvType::FilterBackprop>(
    SYCLConv2DParams const& params, int const window, int const stride,
    int const channel_vector, int const feature_vector) {
  return false;
}
template <typename T, ConvType CType, int tile_rows, int tile_cols,
          int channel_vector_width, int feature_vector_width, bool use_fast_div,
          int window_rows, int window_cols, int stride>
inline bool launch_tiled(Eigen::SyclDevice const& device, T* const output,
                         T const* const input, T const* const filter,
                         SYCLConv2DParams const& params) {
  using Functor =
      Conv2DTiledSYCL<T, CType, tile_rows, tile_cols, channel_vector_width,
                      feature_vector_width, use_fast_div, window_rows,
                      window_cols, stride>;
  static constexpr auto read_mode = Functor::read_mode;
  static constexpr auto write_mode = Functor::write_mode;
  static constexpr auto max_threads = 2048 * 256;
  using Index = int;
  const Index output_size =
      tiled_output_size<CType, tile_rows, tile_cols, channel_vector_width,
                        feature_vector_width>::get(params);
  const Index workgroup_size = device.maxSyclThreadsPerBlock();
  const Index n_threads = std::max(
      RoundUpToNearestMultiple(output_size, workgroup_size), max_threads);

  auto input_buffer = device.get_sycl_buffer(input);
  auto filter_buffer = device.get_sycl_buffer(filter);
  auto output_buffer = device.get_sycl_buffer(output);
  auto kernel_params = get_kernel_params<CType>(params);

  auto event = device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
    auto input_access = input_buffer.template get_access<read_mode>(cgh);
    auto filter_access = filter_buffer.template get_access<read_mode>(cgh);
    auto output_access = output_buffer.template get_access<write_mode>(cgh);

    Functor conv(kernel_params, input_access, filter_access, output_access);

    cgh.parallel_for(cl::sycl::range<1>(n_threads), conv);
  });
  event.wait();
  return true;
}
template <typename T, ConvType CType>
struct LaunchConv2DTiled {
  using Index = int;

  static bool launch(Eigen::SyclDevice const& device, T* const output,
                     T const* const input, T const* const filter,
                     SYCLConv2DParams const& params) {
#define LAUNCH_TILED_CONV(tile_row, tile_col, channel_vector_width,           \
                          feature_vector_width, use_fast_div, window, stride) \
  return launch_tiled<T, CType, tile_row, tile_col, channel_vector_width,     \
                      feature_vector_width, use_fast_div, window, window,     \
                      stride>(device, output, input, filter, params);
#define LAUNCH_IF_MATCH(params, window, stride, tile_row, tile_col,       \
                        channel_vector, feature_vector)                   \
  if (use_static_conv<CType>(params, window, stride, channel_vector,      \
                             feature_vector)) {                           \
    LAUNCH_TILED_CONV(tile_row, tile_col, channel_vector, feature_vector, \
                      false, window, stride)                              \
  }
    // clang-format off
#ifdef SNN_ARM
    LAUNCH_IF_MATCH(params, 3, 1, 1, 4, 1, 1)
    LAUNCH_IF_MATCH(params, 3, 2, 2, 4, 1, 4)
#endif
    if(CType == ConvType::Forward) {
      LAUNCH_IF_MATCH(params, 1, 2, 1, 2, 1, 4)
      LAUNCH_IF_MATCH(params, 1, 2, 1, 2, 1, 1)
      LAUNCH_IF_MATCH(params, 3, 2, 2, 2, 1, 4)
      LAUNCH_IF_MATCH(params, 3, 2, 2, 2, 1, 1)
    }
    if(CType == ConvType::InputBackprop) {
      LAUNCH_IF_MATCH(params, 1, 2, 2, 2, 1, 4)
      LAUNCH_IF_MATCH(params, 1, 2, 2, 2, 1, 1)
      LAUNCH_IF_MATCH(params, 3, 2, 2, 4, 1, 2)
      LAUNCH_IF_MATCH(params, 3, 1, 3, 4, 1, 4)
    }
    LAUNCH_IF_MATCH(params, 3, 1, 2, 2, 1, 4)
    LAUNCH_IF_MATCH(params, 3, 1, 3, 4, 1, 1)
    LAUNCH_IF_MATCH(params, 5, 1, 2, 2, 1, 2)
    LAUNCH_IF_MATCH(params, 5, 1, 2, 4, 1, 1)
    LAUNCH_IF_MATCH(params, 1, 1, 2, 2, 1, 4)
    LAUNCH_IF_MATCH(params, 1, 1, 2, 2, 1, 1)

    return false;
    // clang-format on
#undef LAUNCH_IF_MATCH
#undef LAUNCH_TILED_CONV
  }
};
}  // namespace direct_tiled
template <typename T, typename backend_type, ConvType CType>
struct Launcher<T, backend_type, algorithm::direct_tiled, CType> final
    : public direct_tiled::LaunchConv2DTiled<T, CType> {};
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_NAIVE_SYCL_H_
