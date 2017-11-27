#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_H_

#include "tensorflow/core/kernels/conv_ops_winograd_sycl_kernels.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
namespace functor {
template <typename T, int C1 = 1, int C2 = 1>
struct BatchMatmul {
  using TensorShape = Eigen::DSizes<Eigen::DenseIndex, 3>;
  using TensorType = Eigen::Tensor<T, 3, Eigen::RowMajor, Eigen::DenseIndex>;
  using Tensor = Eigen::TensorMap<TensorType, Eigen::Aligned>;
  using ConstTensorType =
      Eigen::Tensor<T const, 3, Eigen::RowMajor, Eigen::DenseIndex>;
  using ConstTensor = Eigen::TensorMap<ConstTensorType, Eigen::Aligned>;
  using ContractDims = Eigen::IndexPairList<Eigen::type2indexpair<C1, C2>>;

  void operator()(Eigen::SyclDevice const& d, T const* const x_ptr,
                  T const* const y_ptr, T* const z_ptr, const int batches,
                  const int m, const int k, const int n) {
    TensorShape x_shape, y_shape;

    x_shape[0] = batches;
    y_shape[0] = batches;
    if (C1 == 1) {
      x_shape[1] = m;
      x_shape[2] = k;
    } else {
      x_shape[1] = k;
      x_shape[2] = m;
    }
    if (C2 == 1) {
      y_shape[1] = n;
      y_shape[2] = k;
    } else {
      y_shape[1] = k;
      y_shape[2] = n;
    }
    TensorShape z_shape{batches, m, n};

    ConstTensor in_x{x_ptr, x_shape};
    ConstTensor in_y{y_ptr, y_shape};
    Tensor out{z_ptr, z_shape};

    for (int i = 0; i < batches; ++i) {
      auto x = in_x.template chip<0>(i);
      auto y = in_y.template chip<0>(i);
      auto z = out.template chip<0>(i);
      z.device(d) = x.contract(y, ContractDims{});
    }
  }
};
/*
template <typename T, int C1 = 1, int C2 = 1>
struct BatchMatmul {
  using TensorType = Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>;
  using Tensor = Eigen::TensorMap<TensorType, Eigen::Aligned>;
  using TensorShape = Eigen::DSizes<Eigen::DenseIndex, 2>;
  using ConstTensorType =
      Eigen::Tensor<T const, 2, Eigen::RowMajor, Eigen::DenseIndex>;
  using ConstTensor = Eigen::TensorMap<ConstTensorType, Eigen::Aligned>;
  using ContractDims = Eigen::IndexPairList<Eigen::type2indexpair<C1, C2>>;

  void operator()(Eigen::SyclDevice const& d, T const* const in_x,
                  T const* const in_y, T* const out, const int batches,
                  const int m, const int k, const int n) {
    const int x_size = m * k;
    const int y_size = k * n;
    const int z_size = m * n;
    TensorShape x_shape, y_shape;

    if (C1 == 1) {
      x_shape[0] = m;
      x_shape[1] = k;
    } else {
      x_shape[0] = k;
      x_shape[1] = m;
    }
    if (C2 == 1) {
      y_shape[0] = n;
      y_shape[1] = k;
    } else {
      y_shape[0] = k;
      y_shape[1] = n;
    }
    TensorShape z_shape{m, n};

    for (int i = 0; i < batches; ++i) {
      T const* const x_ptr = in_x + i * x_size;
      T const* const y_ptr = in_y + i * y_size;
      T* const z_ptr = out + i * z_size;
      ConstTensor x{x_ptr, x_shape};
      ConstTensor y{y_ptr, y_shape};
      Tensor z{z_ptr, z_shape};
      z.device(d) = x.contract(y, ContractDims{});
    }
  }
};
*/
}  // namespace functor

template <typename T, int M, int N, int R, int S, ConvType CType>
struct LaunchMatmulWinograd {
  using Index = int;
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;

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
    launch_transform<functor::ExtractInputTiles<T, M, N, R, S, CType>>(
        device, input, in_transform, in_transform_items, params, n_tiles);

    size_t const fil_transform_bytes =
        A * B * params.channels_ * params.features_ * sizeof(T);
    T* const fil_transform =
        static_cast<T*>(device.allocate_temp(fil_transform_bytes));
    const Index fil_transform_items = params.features_ * params.channels_;
    launch_transform<functor::ExtractKernelTiles<T, M, N, R, S, CType>>(
        device, filter, fil_transform, fil_transform_items, params, n_tiles);

    size_t const inter_bytes = A * B * n_tiles * params.features_ * sizeof(T);
    T* const intermediate = static_cast<T*>(device.allocate_temp(inter_bytes));
    functor::BatchMatmul<T, 1, 1>()(device, fil_transform, in_transform,
                                    intermediate, A * B, params.features_,
                                    params.channels_, n_tiles);

    device.deallocate_temp(fil_transform);
    device.deallocate_temp(in_transform);

    const Index n_out_items = n_tiles * params.features_;
    launch_transform<functor::ExtractOutputTiles<T, M, N, R, S, CType>>(
        device, intermediate, output, n_out_items, params, n_tiles);

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
    launch_transform<functor::ExtractInputTiles<T, M, N, R, S, CType>>(
        device, input, in_transform, in_transform_items, params, n_tile_rows,
        n_tile_cols);

    size_t const fil_transform_bytes =
        A * B * n_tiles * params.features_ * sizeof(T);
    T* const fil_transform =
        static_cast<T*>(device.allocate_temp(fil_transform_bytes));
    const Index fil_transform_items = params.features_ * n_tiles;
    launch_transform<functor::ExtractKernelTiles<T, M, N, R, S, CType>>(
        device, filter, fil_transform, fil_transform_items, params, n_tiles);

    size_t const n_inter_bytes =
        A * B * params.channels_ * params.features_ * sizeof(T);
    T* const intermediate = static_cast<T*>(device.allocate_temp(n_inter_bytes));
    functor::BatchMatmul<T, 0, 0>()(device, in_transform, fil_transform,
                                    intermediate, A * B, params.channels_,
                                    n_tiles, params.features_);

    device.deallocate_temp(fil_transform);
    device.deallocate_temp(in_transform);

    const Index out_transform_items = params.channels_ * params.features_;
    launch_transform<functor::ExtractOutputTiles<T, M, N, R, S, CType>>(
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
