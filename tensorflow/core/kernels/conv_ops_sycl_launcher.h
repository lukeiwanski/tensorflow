#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_SYCL_LAUNCHER_H_
#define TENSORFLOW_KERNELS_CONV_OPS_SYCL_LAUNCHER_H_

#include "tensorflow/core/kernels/conv_ops_sycl_common.h"

#include "tensorflow/core/kernels/conv_ops_sycl_selectors.h"

#include "tensorflow/core/kernels/conv_ops_direct_sycl.h"
#include "tensorflow/core/kernels/conv_ops_direct_tiled_sycl.h"
#include "tensorflow/core/kernels/conv_ops_im2col_sycl.h"
#include "tensorflow/core/kernels/conv_ops_winograd_sycl.h"

namespace tensorflow {
template <typename T, typename backend_type, algorithm Algo, ConvType CType>
struct Launcher;
template <typename T, typename backend_type>
struct Launcher<T, backend_type, algorithm::matmul, ConvType::Forward> {
  static bool launch(backend_type const& backend, T* const output,
                     T const* const input, T const* const filter,
                     SYCLConv2DParams const& params) {
    int conv_width = params.batch_ * params.out_rows_ * params.out_cols_;

    sycl_conv::launch_matmul<false, false>(backend, input, filter, output,
                                           static_cast<T>(0), conv_width,
                                           params.channels_, params.features_);
    return true;
  }
};
template <typename T, typename backend_type>
struct Launcher<T, backend_type, algorithm::matmul, ConvType::InputBackprop> {
  static bool launch(backend_type const& backend, T* const output,
                     T const* const input, T const* const filter,
                     SYCLConv2DParams const& params) {
    int conv_width = params.batch_ * params.in_rows_ * params.in_cols_;

    sycl_conv::launch_matmul<false, true>(backend, input, filter, output,
                                          static_cast<T>(0), conv_width,
                                          params.features_, params.channels_);
    return true;
  }
};
template <typename T, typename backend_type>
struct Launcher<T, backend_type, algorithm::matmul, ConvType::FilterBackprop> {
  static bool launch(backend_type const& backend, T* const output,
                     T const* const input, T const* const filter,
                     SYCLConv2DParams const& params) {
    int conv_width = params.batch_ * params.in_rows_ * params.in_cols_;

    sycl_conv::launch_matmul<true, false>(backend, input, filter, output,
                                          static_cast<T>(0), params.channels_,
                                          conv_width, params.features_);
    return true;
  }
};
template <typename T, typename backend_type, ConvType CType>
struct Launcher<T, backend_type, algorithm::not_supported, CType> {
  static bool launch(backend_type const& backend, T* const output,
                     T const* const input, T const* const filter,
                     SYCLConv2DParams const& params) {
    LOG(WARNING) << "Convolution algorithm not supported. The results will not "
                    "be computed.";
    return false;
  }
};
/*
// TODO(jwlawson): Add backend interface and propagate across all launchers.
// The aim of the backend is to provide an interface to plug in any GEMM
// library, rather than relying on Eigen. The backend would provide a device
// pointer type, and methods to get the sycl buffers linked to those
// pointers, as well as providing methods to queue cgh tasks and perform GEMM
// and batched GEMM.
static inline void launch_conv2d(backend_type const& backend,
                                 backend_type::device_ptr<const T> const input,
                                 backend_type::device_ptr<const T> const filter,
                                 SYCLConv2DParams const& params,
                                 backend_type::device_ptr<T> const output,
                                 algorithm_selector& selector) {
*/
template <typename T, ConvType CType, typename backend_type>
static inline void launch_conv2d_nchw(backend_type const& backend,
                                      T const* const input,
                                      T const* const filter,
                                      SYCLConv2DParams& params, T* const output,
                                      algorithm_selector& selector) {
  LaunchNCHWConv2DKernel<T, CType>::launch(backend, output, input, filter,
                                           params);
}
template <typename T, ConvType CType, typename backend_type>
static inline void launch_conv2d(backend_type const& backend,
                                 T const* const input, T const* const filter,
                                 SYCLConv2DParams& params, T* const output,
                                 algorithm_selector& selector) {
  algorithm algo = selector.get_selection(params);
#define CALL_LAUNCHER(result, selected)                        \
  result = Launcher<T, backend_type, selected, CType>::launch( \
      backend, output, input, filter, params);
#define CASE(result, selected)                  \
  case algorithm::selected:                     \
    CALL_LAUNCHER(result, algorithm::selected); \
    break;

  bool result;
  switch (algo) {
    CASE(result, matmul);
    CASE(result, winograd_3x1);
    CASE(result, winograd_1x3);
    CASE(result, winograd_3x3);
    CASE(result, im2col);
    CASE(result, direct);
    CASE(result, direct_tiled);
    CASE(result, not_supported);
  }
#undef CASE
  if (!result) {
    VLOG(2) << "Fall back to using direct convolution which does not require "
               "allocations.";
    CALL_LAUNCHER(result, algorithm::direct);
  }
#undef CALL_LAUNCHER
}
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_LAUNCHER_H_
