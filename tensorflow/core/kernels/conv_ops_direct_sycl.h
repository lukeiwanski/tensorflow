#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_NAIVE_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_NAIVE_SYCL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_ops_sycl_common.h"

#include "tensorflow/core/kernels/conv_ops_direct_sycl_kernels.h"
#include "tensorflow/core/kernels/conv_ops_direct_sycl_nchw_kernels.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
namespace direct {
template <ConvType CType>
inline SYCLConv2DParams get_kernel_params(SYCLConv2DParams params) {
  return params;
}
template <>
inline SYCLConv2DParams get_kernel_params<ConvType::InputBackprop>(
    SYCLConv2DParams params) {
  std::swap(params.channels_, params.features_);
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
template <ConvType CType>
inline size_t get_output_size(SYCLConv2DParams const& params);
template <>
inline size_t get_output_size<ConvType::Forward>(
    SYCLConv2DParams const& params) {
  return params.batch_ * params.out_rows_ * params.out_cols_ * params.features_;
}
template <>
inline size_t get_output_size<ConvType::InputBackprop>(
    SYCLConv2DParams const& params) {
  return params.batch_ * params.in_rows_ * params.in_cols_ * params.channels_;
}
template <>
inline size_t get_output_size<ConvType::FilterBackprop>(
    SYCLConv2DParams const& params) {
  return params.window_rows_ * params.window_cols_ * params.channels_ *
         params.features_;
}
template <ConvType CType>
inline bool no_fast_div(SYCLConv2DParams const& params);
template <>
inline bool no_fast_div<ConvType::Forward>(SYCLConv2DParams const& params) {
  return params.features_ == 1 || params.out_rows_ == 1 ||
         params.out_cols_ == 1;
}
template <>
inline bool no_fast_div<ConvType::InputBackprop>(
    SYCLConv2DParams const& params) {
  return params.features_ == 1 || params.in_rows_ == 1 || params.in_cols_ == 1;
}
template <>
inline bool no_fast_div<ConvType::FilterBackprop>(
    SYCLConv2DParams const& params) {
  return params.features_ == 1 || params.channels_ == 1 ||
         params.out_cols_ == 1;
}
template <ConvType>
inline bool use_static_conv(SYCLConv2DParams const& params, int const window,
                            int const stride) {
  return (params.window_cols_ == window && params.window_rows_ == window &&
          params.stride_rows_ == stride && params.stride_cols_ == stride);
}
template <typename T, ConvType CType, bool use_fast_div, int window, int stride>
inline bool launch_direct(Eigen::SyclDevice const& device, T* const output,
                          T const* const input, T const* const filter,
                          SYCLConv2DParams const& params) {
  using Functor = Conv2DSYCL<T, CType, use_fast_div, window, stride>;
  static constexpr auto read_mode = Functor::read_mode;
  static constexpr auto write_mode = Functor::write_mode;
  using Index = int;
  static constexpr Index max_threads = 2048 * 256;
  const Index output_size = get_output_size<CType>(params);
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

    Functor conv(output_size, kernel_params, input_access, filter_access,
                 output_access);

    cgh.parallel_for(cl::sycl::range<1>(n_threads), conv);
  });
  event.wait();
  return true;
}
template <typename T, ConvType CType>
struct LaunchConv2DKernel {
  using Index = int;

  static bool launch(Eigen::SyclDevice const& device, T* const output,
                     T const* const input, T const* const filter,
                     SYCLConv2DParams const& params) {
#define LAUNCH_STATIC_CONV(use_fast_div, window, stride)        \
  return launch_direct<T, CType, use_fast_div, window, stride>( \
      device, output, input, filter, params);
#define LAUNCH_DEFAULT_CONV(use_fast_div) LAUNCH_STATIC_CONV(use_fast_div, 0, 0)
#define LAUNCH_CONV(params, use_fast_div)            \
  if (use_static_conv<CType>(params, 1, 1)) {        \
    LAUNCH_STATIC_CONV(use_fast_div, 1, 1)           \
  } else if (use_static_conv<CType>(params, 3, 1)) { \
    LAUNCH_STATIC_CONV(use_fast_div, 3, 1)           \
  } else if (use_static_conv<CType>(params, 3, 2)) { \
    LAUNCH_STATIC_CONV(use_fast_div, 3, 2)           \
  } else if (use_static_conv<CType>(params, 5, 1)) { \
    LAUNCH_STATIC_CONV(use_fast_div, 5, 1)           \
  } else if (use_static_conv<CType>(params, 5, 2)) { \
    LAUNCH_STATIC_CONV(use_fast_div, 5, 2)           \
  } else {                                           \
    LAUNCH_DEFAULT_CONV(use_fast_div)                \
  }

    auto kernel_params = get_kernel_params<CType>(params);
    if (no_fast_div<CType>(kernel_params)) {
      LAUNCH_CONV(params, false);
    } else {
      LAUNCH_CONV(params, true);
    }
    return false;
#undef LAUNCH_CONV
#undef LAUNCH_DEFAULT_CONV
#undef LAUNCH_STATIC_CONV
  }
};

}  // namespace direct

template <typename T, ConvType CType>
struct LaunchNCHWConv2DKernel {
  using Functor = direct::Conv2DNCHW<T, CType>;
  static constexpr auto read_mode = Functor::read_mode;
  static constexpr auto write_mode = Functor::write_mode;
  using Index = int;

  static bool launch(Eigen::SyclDevice const& device, T* const output,
                     T const* const input, T const* const filter,
                     SYCLConv2DParams const& params) {
    const Index output_size = direct::get_output_size<CType>(params);
    const Index workgroup_size = device.maxSyclThreadsPerBlock();
    const Index n_threads =
        RoundUpToNearestMultiple(output_size, workgroup_size);

    auto input_buffer = device.get_sycl_buffer(input);
    auto filter_buffer = device.get_sycl_buffer(filter);
    auto output_buffer = device.get_sycl_buffer(output);
    auto kernel_params = direct::get_kernel_params<CType>(params);

#define LAUNCH_STATIC_CONV(use_fast_div, window, stride)                     \
  device.sycl_queue().submit([&](cl::sycl::handler& cgh) {                   \
    auto input_access = input_buffer.template get_access<read_mode>(cgh);    \
    auto filter_access = filter_buffer.template get_access<read_mode>(cgh);  \
    auto output_access = output_buffer.template get_access<write_mode>(cgh); \
                                                                             \
    direct::Conv2DNCHW<T, CType, use_fast_div, window, stride> conv(         \
        output_size, kernel_params, input_access, filter_access,             \
        output_access);                                                      \
                                                                             \
    cgh.parallel_for(cl::sycl::range<1>(n_threads), conv);                   \
  });
#define LAUNCH_DEFAULT_CONV(use_fast_div) LAUNCH_STATIC_CONV(use_fast_div, 0, 0)
#define USE_STATIC_CONV(params, window, stride)                      \
  (params.window_cols_ == window && params.window_rows_ == window && \
   params.stride_rows_ == stride && params.stride_cols_ == stride)
#define LAUNCH_CONV(params, use_fast_div)     \
  if (USE_STATIC_CONV(params, 1, 1)) {        \
    LAUNCH_STATIC_CONV(use_fast_div, 1, 1)    \
  } else if (USE_STATIC_CONV(params, 3, 1)) { \
    LAUNCH_STATIC_CONV(use_fast_div, 3, 1)    \
  } else if (USE_STATIC_CONV(params, 3, 2)) { \
    LAUNCH_STATIC_CONV(use_fast_div, 3, 2)    \
  } else if (USE_STATIC_CONV(params, 5, 1)) { \
    LAUNCH_STATIC_CONV(use_fast_div, 5, 1)    \
  } else if (USE_STATIC_CONV(params, 5, 2)) { \
    LAUNCH_STATIC_CONV(use_fast_div, 5, 2)    \
  } else {                                    \
    LAUNCH_DEFAULT_CONV(use_fast_div)         \
  }

    if (direct::no_fast_div<CType>(kernel_params)) {
      LAUNCH_CONV(params, false);
    } else {
      LAUNCH_CONV(params, true);
    }
    return true;
#undef LAUNCH_CONV
#undef USE_STATIC_CONV
#undef LAUNCH_DEFAULT_CONV
#undef LAUNCH_STATIC_CONV
  }
};
template <typename T, typename backend_type, ConvType CType>
struct Launcher<T, backend_type, algorithm::direct, CType> final
    : public direct::LaunchConv2DKernel<T, CType> {};
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_NAIVE_SYCL_H_
