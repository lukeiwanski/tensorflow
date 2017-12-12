#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_H_

#include "tensorflow/core/kernels/conv_ops_winograd_sycl_kernels.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;

template <typename T, int M, int N, int R, int S, ConvType CType>
struct LaunchMatmulWinograd {
  using Index = int;
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  using InputTransform = functor::ExtractInputTiles<T, M, N, R, S, CType>;
  using FilterTransform = functor::ExtractKernelTiles<T, M, N, R, S, CType>;
  using OutputTransform = functor::ExtractOutputTiles<T, M, N, R, S, CType>;

  static bool launch(Eigen::SyclDevice const& device, T* const output,
                     T const* const input, T const* const filter,
                     SYCLConv2DParams& params) {
    // NOTE(jwlawson): We could specialise the launcher to only include this for
    // the input backprop, however the rest of this function is the same between
    // the cases and I would prefer to have less code duplication.
    if (CType == ConvType::InputBackprop) {
      std::swap(params.channels_, params.features_);
      std::swap(params.in_rows_, params.out_rows_);
      std::swap(params.in_cols_, params.out_cols_);
      // We need to change the padding from input padding to output padding for
      // the winograd matmul kernel. pad_out = filt_size - 1 - pad_in
      params.pad_rows_ = params.window_rows_ - 1 - params.pad_rows_;
      params.pad_cols_ = params.window_cols_ - 1 - params.pad_cols_;
    }
    const Index n_tile_rows = RoundRatioUpAboveZero(params.out_rows_, M);
    const Index n_tile_cols = RoundRatioUpAboveZero(params.out_cols_, N);
    const Index n_tiles = params.batch_ * n_tile_rows * n_tile_cols;

    size_t const in_transform_bytes =
        A * B * n_tiles * params.channels_ * sizeof(T);
    T* const in_transform =
        static_cast<T*>(device.allocate_temp(in_transform_bytes));
    const Index in_transform_items = n_tiles * params.channels_;
    sycl_conv::launch_transform<InputTransform>(
        device, input, in_transform, in_transform_items, params, n_tiles);

    size_t const fil_transform_bytes =
        A * B * params.channels_ * params.features_ * sizeof(T);
    T* const fil_transform =
        static_cast<T*>(device.allocate_temp(fil_transform_bytes));
    const Index fil_transform_items = params.features_ * params.channels_;
    sycl_conv::launch_transform<FilterTransform>(
        device, filter, fil_transform, fil_transform_items, params, n_tiles);

    size_t const inter_bytes = A * B * n_tiles * params.features_ * sizeof(T);
    T* const intermediate = static_cast<T*>(device.allocate_temp(inter_bytes));
    sycl_conv::launch_batch_matmul<false, true>(
        device, fil_transform, in_transform, intermediate, A * B,
        params.features_, params.channels_, n_tiles);

    device.deallocate_temp(fil_transform);
    device.deallocate_temp(in_transform);

    const Index n_out_items = n_tiles * params.features_;
    sycl_conv::launch_transform<OutputTransform>(device, intermediate, output,
                                                 n_out_items, params, n_tiles);

    device.deallocate_temp(intermediate);
    return true;
  }
};
template <typename T, int M, int N, int R, int S>
struct LaunchMatmulWinograd<T, M, N, R, S, ConvType::FilterBackprop> {
  using Index = int;
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  static constexpr auto CType = ConvType::FilterBackprop;
  using InputTransform = functor::ExtractInputTiles<T, M, N, R, S, CType>;
  using FilterTransform = functor::ExtractKernelTiles<T, M, N, R, S, CType>;
  using OutputTransform = functor::ExtractOutputTiles<T, M, N, R, S, CType>;

  static bool launch(Eigen::SyclDevice const& device, T* const output,
                     T const* const input, T const* const filter,
                     SYCLConv2DParams& params) {
    // Map the input dimensions to those expected in the convolution kernel.
    const Index window_rows =
        params.out_rows_ * params.stride_rows_ - (params.stride_rows_ - 1);
    const Index window_cols =
        params.out_cols_ * params.stride_cols_ - (params.stride_cols_ - 1);
    params.out_rows_ = params.window_rows_;
    params.out_cols_ = params.window_cols_;
    params.window_rows_ = window_rows;
    params.window_cols_ = window_cols;

    const Index n_tile_rows = RoundRatioUpAboveZero(params.window_rows_, R);
    const Index n_tile_cols = RoundRatioUpAboveZero(params.window_cols_, S);
    const Index n_tiles = params.batch_ * n_tile_rows * n_tile_cols;

    const size_t in_transform_bytes =
        A * B * n_tiles * params.channels_ * sizeof(T);
    T* const in_transform =
        static_cast<T*>(device.allocate_temp(in_transform_bytes));
    const Index in_transform_items = n_tiles * params.channels_;
    sycl_conv::launch_transform<InputTransform>(device, input, in_transform,
                                                in_transform_items, params,
                                                n_tile_rows, n_tile_cols);

    size_t const fil_transform_bytes =
        A * B * n_tiles * params.features_ * sizeof(T);
    T* const fil_transform =
        static_cast<T*>(device.allocate_temp(fil_transform_bytes));
    const Index fil_transform_items = params.features_ * n_tiles;
    sycl_conv::launch_transform<FilterTransform>(
        device, filter, fil_transform, fil_transform_items, params, n_tiles);

    size_t const n_inter_bytes =
        A * B * params.channels_ * params.features_ * sizeof(T);
    T* const intermediate =
        static_cast<T*>(device.allocate_temp(n_inter_bytes));
    sycl_conv::launch_batch_matmul<true, false>(
        device, in_transform, fil_transform, intermediate, A * B,
        params.channels_, n_tiles, params.features_);

    device.deallocate_temp(fil_transform);
    device.deallocate_temp(in_transform);

    const Index out_transform_items = params.channels_ * params.features_;
    sycl_conv::launch_transform<OutputTransform>(
        device, intermediate, output, out_transform_items, params, n_tiles);

    device.deallocate_temp(intermediate);
    return true;
  }
};
}  // namespace tensorflow
#include "tensorflow/core/kernels/conv_ops_winograd_sycl_impl.h"
namespace tensorflow {
template <typename T, typename backend_type, ConvType CType>
struct Launcher<T, backend_type, algorithm::winograd_3x3, CType> final
    : public LaunchMatmulWinograd<T, 2, 2, 3, 3, CType> {};
template <typename T, typename backend_type, ConvType CType>
struct Launcher<T, backend_type, algorithm::winograd_3x1, CType> final
    : public LaunchMatmulWinograd<T, 2, 1, 3, 1, CType> {};
template <typename T, typename backend_type, ConvType CType>
struct Launcher<T, backend_type, algorithm::winograd_1x3, CType> final
    : public LaunchMatmulWinograd<T, 1, 2, 1, 3, CType> {};

template <typename T, typename backend_type>
struct Launcher<T, backend_type, algorithm::winograd_3x3,
                ConvType::FilterBackprop>
    final
    : public LaunchMatmulWinograd<T, 3, 3, 2, 2, ConvType::FilterBackprop> {};
template <typename T, typename backend_type>
struct Launcher<T, backend_type, algorithm::winograd_3x1,
                ConvType::FilterBackprop>
    final
    : public LaunchMatmulWinograd<T, 3, 1, 2, 1, ConvType::FilterBackprop> {};
template <typename T, typename backend_type>
struct Launcher<T, backend_type, algorithm::winograd_1x3,
                ConvType::FilterBackprop>
    final
    : public LaunchMatmulWinograd<T, 1, 3, 1, 2, ConvType::FilterBackprop> {};
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_H_
