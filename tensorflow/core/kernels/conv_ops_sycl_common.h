#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_SYCL_COMMON_H_
#define TENSORFLOW_KERNELS_CONV_OPS_SYCL_COMMON_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
/**
 * Helper function to provide the ratio of two integers, always rounded up. If
 * the numerator is negative then we assume that the rounded ratio will be
 * zero, otherwise we need to ensure that the value is rounded up rather
 * than down.
 */
template <
    typename IntegerType,
    typename std::enable_if<std::is_signed<IntegerType>::value, int>::type = 0>
inline TF_ATTRIBUTE_ALWAYS_INLINE IntegerType
RoundRatioUpAboveZero(const IntegerType num, const IntegerType div) {
  static_assert(std::is_integral<IntegerType>::value,
                "RoundRatioUpAboveZero is only valid for integral types");
  return num < 0 ? 0 : (num % div != 0 ? num / div + 1 : num / div);
}
template <typename IntegerType,
          typename std::enable_if<std::is_unsigned<IntegerType>::value>::type* =
              nullptr>
inline TF_ATTRIBUTE_ALWAYS_INLINE IntegerType
RoundRatioUpAboveZero(const IntegerType num, const IntegerType div) {
  static_assert(std::is_integral<IntegerType>::value,
                "RoundRatioUpAboveZero is only valid for integral types");
  return num % div != 0 ? num / div + 1 : num / div;
}
/**
 * Helper function to provide the ratio of two integers, always rounded up.
 */
template <typename IntegerType>
inline TF_ATTRIBUTE_ALWAYS_INLINE IntegerType
RoundRatioUp(const IntegerType num, const IntegerType div) {
  static_assert(std::is_integral<IntegerType>::value,
                "RoundRatioUp is only valid for integral types");
  IntegerType quotient = num / div;
  IntegerType additive = num % div == 0 ? 0 : 1;
  return num < 0 ? quotient : quotient + additive;
}
/**
 * Helper function to round up an integral value to the nearest multiple of a
 * given multiplier.
 */
template <typename IntegerType>
inline TF_ATTRIBUTE_ALWAYS_INLINE IntegerType
RoundUpToNearestMultiple(IntegerType val, const IntegerType multiplier) {
  static_assert(std::is_integral<IntegerType>::value,
                "RoundUpToNearestMultiple is only valid for integral types");
  const IntegerType diff = val % multiplier;
  if (diff > 0) {
    val += (multiplier - diff);
  }
  return val;
}
/** Enum to allow specialisations for different convolution algorithms. */
enum class ConvType {
  Forward,
  InputBackprop,
  FilterBackprop,
};
enum class DataLayout {
  NHWC,
  NCHW,
};
template <DataLayout D>
inline int TensorIndex(int batch, int row, int col, int channel, int n_rows,
                int n_cols, int n_channels);
template <>
inline int TensorIndex<DataLayout::NHWC>(int batch, int row, int col, int channel,
                                  int n_rows, int n_cols, int n_channels) {
  return ((batch * n_rows + row) * n_cols + col) * n_channels + channel;
}
template <>
inline int TensorIndex<DataLayout::NCHW>(int batch, int row, int col, int channel,
                                  int n_rows, int n_cols, int n_channels) {
  return ((batch * n_channels + channel) * n_rows + row) * n_cols + col;
}
/** The different algorithms supported to compute convolutions. */
enum class algorithm {
  matmul,
  winograd_1x3,
  winograd_3x1,
  winograd_3x3,
  im2col,
  direct,
  direct_tiled,
  not_supported,
};
template <typename T, typename backend_type, algorithm Algo, ConvType CType>
struct Launcher;

struct SYCL2DWindow {
  using Index = int;

  const Index rstart;
  const Index rend;
  const Index firstr;

  const Index cstart;
  const Index cend;
  const Index firstc;

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
struct SYCLOutputWindow {
  using Index = int;

  const Index rsize;
  const Index csize;
  const Index offset;
};
struct SYCLConv2DParams {
  using Index = int;

  /**
   * Note: The _Index template here allows any type of index to be used,
   * provided it can be statically cast to the index type use by the parameters.
   */
  template <typename _Index>
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCLConv2DParams(
      const _Index channels, const _Index features, const _Index batch,
      const _Index in_rows, const _Index in_cols, const _Index window_rows,
      const _Index window_cols, const _Index stride_rows,
      const _Index stride_cols, const _Index out_rows, const _Index out_cols,
      const _Index pad_rows, const _Index pad_cols)
      : channels_{static_cast<Index>(channels)},
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
        pad_cols_{static_cast<Index>(pad_cols)},
        dilation_rows_{1},
        dilation_cols_{1} {}

  Index channels_;
  Index features_;
  Index batch_;

  Index in_rows_;
  Index in_cols_;

  Index window_rows_;
  Index window_cols_;

  Index stride_rows_;
  Index stride_cols_;

  Index out_rows_;
  Index out_cols_;

  Index pad_rows_;
  Index pad_cols_;

  Index dilation_rows_;
  Index dilation_cols_;
};
namespace sycl_conv {
// Ideally we would use a variadic template to capture the arg parameters to
// pass to the functor constructor, however there is a bug in gcc 4.8 which
// prevents the variadic parameter pack from being passed to the cgh lambda. We
// provide a number of functions here to work around this.
template <typename Functor, typename T, typename Index, typename Arg>
static cl::sycl::event launch_transform(Eigen::SyclDevice const& device,
                                        T const* const input,
                                        T* const transform, Index const n_items,
                                        SYCLConv2DParams const& params,
                                        Arg arg) {
  static constexpr auto read_mode = Functor::read_mode;
  static constexpr auto write_mode = Functor::write_mode;

  Index const workgroup_size = device.maxSyclThreadsPerBlock();
  Index const n_threads = RoundUpToNearestMultiple(n_items, workgroup_size);

  auto event = device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
    auto input_access = device.get_sycl_accessor<read_mode>(cgh, input);
    auto transform_access =
        device.get_sycl_accessor<write_mode>(cgh, transform);

    Functor extract_fun(arg, params, input_access, transform_access);
    cgh.parallel_for(cl::sycl::range<1>(n_threads), extract_fun);
  });
  return event;
}
template <typename Functor, typename T, typename Index, typename Arg1,
          typename Arg2>
static cl::sycl::event launch_transform(Eigen::SyclDevice const& device,
                                        T const* const input,
                                        T* const transform, Index const n_items,
                                        SYCLConv2DParams const& params,
                                        Arg1 arg1, Arg2 arg2) {
  static constexpr auto read_mode = Functor::read_mode;
  static constexpr auto write_mode = Functor::write_mode;

  Index const workgroup_size = device.maxSyclThreadsPerBlock();
  Index const n_threads = RoundUpToNearestMultiple(n_items, workgroup_size);

  auto event = device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
    auto input_access = device.get_sycl_accessor<read_mode>(cgh, input);
    auto transform_access =
        device.get_sycl_accessor<write_mode>(cgh, transform);

    Functor extract_fun(arg1, arg2, params, input_access, transform_access);
    cgh.parallel_for(cl::sycl::range<1>(n_threads), extract_fun);
  });
  return event;
}
template <typename Functor, typename T, typename Index, typename Arg1,
          typename Arg2, typename Arg3>
static cl::sycl::event launch_transform(Eigen::SyclDevice const& device,
                                        T const* const input,
                                        T* const transform, Index const n_items,
                                        SYCLConv2DParams const& params,
                                        Arg1 arg1, Arg2 arg2, Arg3 arg3) {
  static constexpr auto read_mode = Functor::read_mode;
  static constexpr auto write_mode = Functor::write_mode;

  Index const workgroup_size = device.maxSyclThreadsPerBlock();
  Index const n_threads = RoundUpToNearestMultiple(n_items, workgroup_size);

  auto event = device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
    auto input_access = device.get_sycl_accessor<read_mode>(cgh, input);
    auto transform_access =
        device.get_sycl_accessor<write_mode>(cgh, transform);

    Functor extract_fun(arg1, arg2, arg3, params, input_access,
                        transform_access);
    cgh.parallel_for(cl::sycl::range<1>(n_threads), extract_fun);
  });
  return event;
}
template <typename Functor, typename T, typename Index, typename Arg1,
          typename Arg2, typename Arg3, typename Arg4>
static cl::sycl::event launch_transform(Eigen::SyclDevice const& device,
                                        T const* const input,
                                        T* const transform, Index const n_items,
                                        SYCLConv2DParams const& params,
                                        Arg1 arg1, Arg2 arg2, Arg3 arg3,
                                        Arg4 arg4) {
  static constexpr auto read_mode = Functor::read_mode;
  static constexpr auto write_mode = Functor::write_mode;

  const Index workgroup_size = device.maxSyclThreadsPerBlock();
  const Index n_threads = RoundUpToNearestMultiple(n_items, workgroup_size);

  auto event = device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
    auto input_access = device.get_sycl_accessor<read_mode>(cgh, input);
    auto transform_access =
        device.get_sycl_accessor<write_mode>(cgh, transform);

    Functor extract_fun(arg1, arg2, arg3, arg4, params, input_access,
                        transform_access);
    cgh.parallel_for(cl::sycl::range<1>(n_threads), extract_fun);
  });
  return event;
}
template <bool trans_lhs, bool trans_rhs, typename T, typename Index>
static void launch_matmul(Eigen::SyclDevice const& device, T const* const lhs,
                          T const* const rhs, T* const output, T const alpha,
                          Index const m, Index const k, Index const n) {
  static constexpr auto lhs_dim = trans_lhs ? 0 : 1;
  static constexpr auto rhs_dim = trans_rhs ? 1 : 0;
  using ConstTensorType =
      Eigen::Tensor<T const, 2, Eigen::RowMajor, Eigen::DenseIndex>;
  using ConstTensor = Eigen::TensorMap<ConstTensorType, Eigen::Aligned>;
  using TensorType = Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>;
  using Tensor = Eigen::TensorMap<TensorType, Eigen::Aligned>;
  using TensorShape = Eigen::DSizes<Eigen::DenseIndex, 2>;
  using ContractDims =
      Eigen::IndexPairList<Eigen::type2indexpair<lhs_dim, rhs_dim>>;

  TensorShape const lhs_shape{trans_lhs ? k : m, trans_lhs ? m : k};
  TensorShape const rhs_shape{trans_rhs ? n : k, trans_rhs ? k : n};
  TensorShape const out_shape{m, n};

  ConstTensor lhs_tensor{lhs, lhs_shape};
  ConstTensor rhs_tensor{rhs, rhs_shape};
  Tensor out_tensor{output, out_shape};

  if (alpha == static_cast<T>(0)) {
    out_tensor.device(device) = lhs_tensor.contract(rhs_tensor, ContractDims{});
  } else {
    out_tensor.device(device) =
        alpha * out_tensor + lhs_tensor.contract(rhs_tensor, ContractDims{});
  }
}
template <bool trans_lhs, bool trans_rhs, typename T, typename Index>
static void launch_batch_matmul(Eigen::SyclDevice const& d,
                                T const* const x_ptr, T const* const y_ptr,
                                T* const z_ptr, Index const batches,
                                Index const m, Index const k, Index const n) {
  Index const x_size = m * k;
  Index const y_size = k * n;
  Index const z_size = m * n;

  for (int i = 0; i < batches; ++i) {
    Index const x_offset = x_size * i;
    Index const y_offset = y_size * i;
    Index const z_offset = z_size * i;
    launch_matmul<trans_lhs, trans_rhs>(d, x_ptr + x_offset, y_ptr + y_offset,
                                        z_ptr + z_offset, static_cast<T>(0), m,
                                        k, n);
  }
}
}  // namespace sycl_conv
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_COMMON_H_
