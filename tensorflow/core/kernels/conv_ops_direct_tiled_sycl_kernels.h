#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_DIRECT_TILED_SYCL_KERNELS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_DIRECT_TILED_SYCL_KERNELS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_ops_sycl_common.h"
#include "tensorflow/core/kernels/conv_ops_sycl_fast_div.h"
#include "tensorflow/core/kernels/conv_ops_sycl_kernel_helpers.h"
#include "tensorflow/core/kernels/conv_ops_sycl_kernel_macros.h"
#include "tensorflow/core/kernels/conv_ops_sycl_param_macros.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
namespace direct_tiled {
struct check_bounds_tag {};
struct mirror_filter_tag {};
template <typename T, int channel_vector, int width>
struct InputRow {
  using Index = int;
  using VecType = cl::sycl::vec<T, channel_vector>;
  template <typename _T>
  inline SNN_ALWAYS_INLINE InputRow(_T const* input, Index const row,
                                    Index const n_rows, Index const col,
                                    Index const n_cols, Index const channel,
                                    Index const n_channels) {
    for (int i = 0; i < width; ++i) {
      Index const idx = (row * n_cols + col + i) * n_channels + channel;
      data[i] = helpers::io::Load<VecType>(input, idx);
    }
  }
  template <typename _T>
  inline SNN_ALWAYS_INLINE InputRow(_T const* input, Index const row,
                                    Index const n_rows, Index const col,
                                    Index const n_cols, Index const channel,
                                    Index const n_channels, check_bounds_tag) {
    for (int i = 0; i < width; ++i) {
      Index const idx = (row * n_cols + col + i) * n_channels + channel;
      data[i] = (col + i < 0 || col + i >= n_cols)
                    ? VecType{static_cast<T>(0)}
                    : helpers::io::Load<VecType>()(input, idx);
    }
  }
  VecType data[width];
};
template <typename T, int channel_vector_width, int feature_vector_width,
          int window_rows, int window_cols>
struct FilterTile {
  using Index = int;
  using VecType = cl::sycl::vec<T, feature_vector_width>;
  template <typename _T>
  inline SNN_ALWAYS_INLINE FilterTile(_T const* const input,
                                      Index const channel,
                                      Index const n_channels,
                                      Index const feature,
                                      Index const n_features) {
    for (int i = 0; i < window_rows; ++i) {
      for (int j = 0; j < window_cols; ++j) {
        for (int ch_v = 0; ch_v < channel_vector_width; ++ch_v) {
          Index const idx =
              ((i * window_cols + j) * n_channels + channel + ch_v) *
                  n_features +
              feature;
          data[i][j][ch_v] = helpers::io::Load<VecType>()(input, idx);
        }
      }
    }
  }
  template <typename _T>
  inline SNN_ALWAYS_INLINE FilterTile(
      _T const* const input, Index const channel, Index const n_channels,
      Index const feature, Index const n_features, mirror_filter_tag) {
    for (int i = 0; i < window_rows; ++i) {
      for (int j = 0; j < window_cols; ++j) {
        SNN_PRAGMA_UNROLL
        for (int ch_v = 0; ch_v < channel_vector_width; ++ch_v) {
          Index const idx =
              ((i * window_cols + j) * n_channels + channel + ch_v) *
                  n_features +
              feature;
          data[window_rows - 1 - i][window_cols - 1 - j][ch_v] =
              helpers::io::Load<VecType>()(input, idx);
        }
      }
    }
  }
  VecType data[window_rows][window_cols][channel_vector_width];
};
template <typename T, int vector_width, int n_out_rows, int n_out_cols>
struct OutputTile {
  using Index = int;
  using VecType = cl::sycl::vec<T, vector_width>;
  template <typename _T>
  inline SNN_ALWAYS_INLINE void write_out(
      _T* output, Index const batch, Index const out_row, Index const n_rows,
      Index const out_col, Index const n_cols, Index const feature,
      Index const n_features) {
    Index const offset =
        ((batch * n_rows + out_row) * n_cols + out_col) * n_features + feature;
    output += offset;
    Index const max_row = cl::sycl::min(n_out_rows, n_rows - out_row);
    Index const max_col = cl::sycl::min(n_out_cols, n_cols - out_col);
    for (Index tile_row = 0; tile_row < max_row; ++tile_row) {
      for (Index tile_col = 0; tile_col < max_col; ++tile_col) {
        Index const idx = (tile_row * n_cols + tile_col) * n_features;
        helpers::io::Store<VecType>()(output, idx, data[tile_row][tile_col]);
      }
    }
  }
  VecType data[n_out_rows][n_out_cols];
};
template <typename T, int in_vector_width, int out_vector_width,
          int n_window_rows, int n_window_cols>
struct FwdAccumulator {};
template <typename T, int in_vector_width, int out_vector_width, int ft_rows,
          int ft_cols>
struct InBkAccumulator {};
template <ConvType CType, typename T, int channel_vector_width,
          int feature_vector_width, int n_window_rows, int n_window_cols>
struct AccumulateOutput {};
template <typename T, int channel_vector_width, int feature_vector_width,
          int n_window_rows, int n_window_cols>
struct AccumulateOutput<ConvType::Forward, T, channel_vector_width,
                        feature_vector_width, n_window_rows, n_window_cols>
    final : public FwdAccumulator<T, channel_vector_width, feature_vector_width,
                                  n_window_rows, n_window_cols> {};
template <typename T, int channel_vector_width, int feature_vector_width,
          int n_window_rows, int n_window_cols>
struct AccumulateOutput<ConvType::InputBackprop, T, channel_vector_width,
                        feature_vector_width, n_window_rows, n_window_cols>
    final
    : public InBkAccumulator<T, channel_vector_width, feature_vector_width,
                             n_window_rows, n_window_cols> {};

template <typename T, int in_vector_width, int ft_rows, int ft_cols>
struct InBkAccumulator<T, 1, in_vector_width, ft_rows, ft_cols> {
  using InVecType = cl::sycl::vec<T, in_vector_width>;
  using OutVecType = cl::sycl::vec<T, 1>;
  OutVecType SNN_ALWAYS_INLINE operator()(
      InVecType input,
      FilterTile<T, 1, in_vector_width, ft_rows, ft_cols> const& filter,
      int const filter_row, int const filter_col, OutVecType initial_val) {
    T res0 = helpers::math::Dot<InVecType>()(
                 input, filter.data[filter_row][filter_col][0]) +
             initial_val.s0();
    return OutVecType{res0};
  }
};
template <typename T, int in_vector_width, int ft_rows, int ft_cols>
struct InBkAccumulator<T, 2, in_vector_width, ft_rows, ft_cols> {
  using InVecType = cl::sycl::vec<T, in_vector_width>;
  using OutVecType = cl::sycl::vec<T, 2>;
  OutVecType SNN_ALWAYS_INLINE operator()(
      InVecType input,
      FilterTile<T, 2, in_vector_width, ft_rows, ft_cols> const& filter,
      int const filter_row, int const filter_col, OutVecType initial_val) {
    T res0 = helpers::math::Dot<InVecType>()(
                 input, filter.data[filter_row][filter_col][0]) +
             initial_val.s0();
    T res1 = helpers::math::Dot<InVecType>()(
                 input, filter.data[filter_row][filter_col][1]) +
             initial_val.s1();
    return OutVecType{res0, res1};
  }
};
template <typename T, int in_vector_width, int ft_rows, int ft_cols>
struct InBkAccumulator<T, 4, in_vector_width, ft_rows, ft_cols> {
  using InVecType = cl::sycl::vec<T, in_vector_width>;
  using OutVecType = cl::sycl::vec<T, 4>;
  OutVecType SNN_ALWAYS_INLINE operator()(
      InVecType input,
      FilterTile<T, 4, in_vector_width, ft_rows, ft_cols> const& filter,
      int const filter_row, int const filter_col, OutVecType initial_val) {
    T res0 = helpers::math::Dot<InVecType>()(
                 input, filter.data[filter_row][filter_col][0]) +
             initial_val.s0();
    T res1 = helpers::math::Dot<InVecType>()(
                 input, filter.data[filter_row][filter_col][1]) +
             initial_val.s1();
    T res2 = helpers::math::Dot<InVecType>()(
                 input, filter.data[filter_row][filter_col][2]) +
             initial_val.s2();
    T res3 = helpers::math::Dot<InVecType>()(
                 input, filter.data[filter_row][filter_col][3]) +
             initial_val.s3();
    return OutVecType{res0, res1, res2, res3};
  }
};

template <typename T, int out_vector_width, int ft_rows, int ft_cols>
struct FwdAccumulator<T, 1, out_vector_width, ft_rows, ft_cols> {
  using InVecType = cl::sycl::vec<T, 1>;
  using OutVecType = cl::sycl::vec<T, out_vector_width>;
  OutVecType SNN_ALWAYS_INLINE operator()(
      InVecType input,
      FilterTile<T, 1, out_vector_width, ft_rows, ft_cols> const& filter,
      int const filter_row, int const filter_col, OutVecType initial_val) {
    return helpers::math::Mad<OutVecType>()(
        OutVecType{input.s0()}, filter.data[filter_row][filter_col][0],
        initial_val);
  }
};
template <typename T, int out_vector_width, int ft_rows, int ft_cols>
struct FwdAccumulator<T, 2, out_vector_width, ft_rows, ft_cols> {
  using InVecType = cl::sycl::vec<T, 2>;
  using OutVecType = cl::sycl::vec<T, out_vector_width>;
  OutVecType SNN_ALWAYS_INLINE operator()(
      InVecType input,
      FilterTile<T, 2, out_vector_width, ft_rows, ft_cols> const& filter,
      int const filter_row, int const filter_col, OutVecType initial_val) {
    OutVecType res1 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s0()}, filter.data[filter_row][filter_col][0],
        initial_val);
    OutVecType res2 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s1()}, filter.data[filter_row][filter_col][1], res1);
    return res2;
  }
};
template <typename T, int out_vector_width, int ft_rows, int ft_cols>
struct FwdAccumulator<T, 4, out_vector_width, ft_rows, ft_cols> {
  using InVecType = cl::sycl::vec<T, 4>;
  using OutVecType = cl::sycl::vec<T, out_vector_width>;
  OutVecType SNN_ALWAYS_INLINE operator()(
      InVecType input,
      FilterTile<T, 4, out_vector_width, ft_rows, ft_cols> const& filter,
      int const filter_row, int const filter_col, OutVecType initial_val) {
    OutVecType res1 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s0()}, filter.data[filter_row][filter_col][0],
        initial_val);
    OutVecType res2 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s1()}, filter.data[filter_row][filter_col][1], res1);
    OutVecType res3 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s2()}, filter.data[filter_row][filter_col][2], res2);
    OutVecType res4 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s3()}, filter.data[filter_row][filter_col][3], res3);
    return res4;
  }
};
template <typename T, int out_vector_width, int ft_rows, int ft_cols>
struct FwdAccumulator<T, 8, out_vector_width, ft_rows, ft_cols> {
  using InVecType = cl::sycl::vec<T, 8>;
  using OutVecType = cl::sycl::vec<T, out_vector_width>;
  OutVecType SNN_ALWAYS_INLINE operator()(
      InVecType input,
      FilterTile<T, 8, out_vector_width, ft_rows, ft_cols> const& filter,
      int const filter_row, int const filter_col, OutVecType initial_val) {
    OutVecType res1 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s0()}, filter.data[filter_row][filter_col][0],
        initial_val);
    OutVecType res2 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s1()}, filter.data[filter_row][filter_col][1], res1);
    OutVecType res3 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s2()}, filter.data[filter_row][filter_col][2], res2);
    OutVecType res4 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s3()}, filter.data[filter_row][filter_col][3], res3);
    OutVecType res5 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s4()}, filter.data[filter_row][filter_col][4], res4);
    OutVecType res6 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s5()}, filter.data[filter_row][filter_col][5], res5);
    OutVecType res7 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s6()}, filter.data[filter_row][filter_col][6], res6);
    OutVecType res8 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s7()}, filter.data[filter_row][filter_col][7], res7);
    return res8;
  }
};
template <typename T, int out_vector_width, int ft_rows, int ft_cols>
struct FwdAccumulator<T, 16, out_vector_width, ft_rows, ft_cols> {
  using InVecType = cl::sycl::vec<T, 16>;
  using OutVecType = cl::sycl::vec<T, out_vector_width>;
  OutVecType SNN_ALWAYS_INLINE operator()(
      InVecType input,
      FilterTile<T, 16, out_vector_width, ft_rows, ft_cols> const& filter,
      int const filter_row, int const filter_col, OutVecType initial_val) {
    OutVecType res1 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s0()}, filter.data[filter_row][filter_col][0],
        initial_val);
    OutVecType res2 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s1()}, filter.data[filter_row][filter_col][1], res1);
    OutVecType res3 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s2()}, filter.data[filter_row][filter_col][2], res2);
    OutVecType res4 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s3()}, filter.data[filter_row][filter_col][3], res3);
    OutVecType res5 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s4()}, filter.data[filter_row][filter_col][4], res4);
    OutVecType res6 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s5()}, filter.data[filter_row][filter_col][5], res5);
    OutVecType res7 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s6()}, filter.data[filter_row][filter_col][6], res6);
    OutVecType res8 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s7()}, filter.data[filter_row][filter_col][7], res7);
    OutVecType res9 = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s8()}, filter.data[filter_row][filter_col][8], res8);
    OutVecType resa = helpers::math::Mad<OutVecType>()(
        OutVecType{input.s9()}, filter.data[filter_row][filter_col][9], res9);
    OutVecType resb = helpers::math::Mad<OutVecType>()(
        OutVecType{input.sA()}, filter.data[filter_row][filter_col][10], resa);
    OutVecType resc = helpers::math::Mad<OutVecType>()(
        OutVecType{input.sB()}, filter.data[filter_row][filter_col][11], resb);
    OutVecType resd = helpers::math::Mad<OutVecType>()(
        OutVecType{input.sC()}, filter.data[filter_row][filter_col][12], resc);
    OutVecType rese = helpers::math::Mad<OutVecType>()(
        OutVecType{input.sD()}, filter.data[filter_row][filter_col][13], resd);
    OutVecType resf = helpers::math::Mad<OutVecType>()(
        OutVecType{input.sE()}, filter.data[filter_row][filter_col][14], rese);
    OutVecType res = helpers::math::Mad<OutVecType>()(
        OutVecType{input.sF()}, filter.data[filter_row][filter_col][15], resf);
    return res;
  }
};
template <int stride, typename T, int channel_vector, int feature_vector,
          int n_window_rows, int n_window_cols, int n_out_rows, int n_out_cols>
inline SNN_ALWAYS_INLINE void convolve_1xw_one_row_fwd(
    InputRow<T, channel_vector,
             (n_out_cols - 1) * stride + n_window_rows> const& input,
    FilterTile<T, channel_vector, feature_vector, n_window_rows,
               n_window_cols> const& filter,
    OutputTile<T, feature_vector, n_out_rows, n_out_cols>& output,
    int const out_row, int const filter_row) {
  for (int out_col = 0; out_col < n_out_cols; ++out_col) {
    int const in_offset = out_col * stride;
    for (int filter_col = 0; filter_col < n_window_cols; ++filter_col) {
      output.data[out_row][out_col] =
          AccumulateOutput<ConvType::Forward, T, channel_vector, feature_vector,
                           n_window_rows, n_window_cols>()(
              input.data[in_offset + filter_col], filter, filter_row,
              filter_col, output.data[out_row][out_col]);
    }
  }
}
template <int stride, typename T, int channel_vector, int feature_vector,
          int n_window_rows, int n_window_cols, int n_out_rows, int n_out_cols>
inline SNN_ALWAYS_INLINE void convolve_1xw_whole_tile_fwd(
    InputRow<T, channel_vector,
             (n_out_cols - 1) * stride + n_window_rows> const& input,
    FilterTile<T, channel_vector, feature_vector, n_window_rows,
               n_window_cols> const& filter,
    OutputTile<T, feature_vector, n_out_rows, n_out_cols>& output,
    int const row_idx) {
  SNN_PRAGMA_UNROLL
  for (int out_row = 0; out_row < n_out_rows; ++out_row) {
    int const filter_row = row_idx - out_row * stride;
    if (filter_row >= 0 && filter_row < n_window_rows) {
      convolve_1xw_one_row_fwd<stride>(input, filter, output, out_row,
                                       filter_row);
    }
  }
}
template <int stride, typename T, int channel_vector, int feature_vector,
          int n_window_rows, int n_window_cols, int n_out_rows, int n_out_cols>
inline SNN_ALWAYS_INLINE void convolve_1xw_one_row_bkin(
    InputRow<T, feature_vector,
             (n_out_cols + n_window_cols - 1) / stride> const& input,
    FilterTile<T, channel_vector, feature_vector, n_window_rows,
               n_window_cols> const& filter,
    OutputTile<T, channel_vector, n_out_rows, n_out_cols>& output,
    int const out_row, int const filter_row, int first_col) {
  assert(first_col >= 0);
  assert(first_col < stride);
  for (int out_col = 0; out_col < n_out_cols; ++out_col) {
    int in_offset = out_col / stride + first_col;
    for (int filter_col = first_col; filter_col < n_window_cols;
         filter_col += stride, ++in_offset) {
      output.data[out_row][out_col] =
          AccumulateOutput<ConvType::InputBackprop, T, channel_vector,
                           feature_vector, n_window_rows, n_window_cols>()(
              input.data[in_offset], filter, filter_row, filter_col,
              output.data[out_row][out_col]);
    }
    first_col--;
    if (first_col < 0) {
      first_col = stride - 1;
    }
  }
}
template <int stride, typename T, int channel_vector, int feature_vector,
          int n_window_rows, int n_window_cols, int n_out_rows, int n_out_cols>
inline SNN_ALWAYS_INLINE void convolve_1xw_whole_tile_bkin(
    InputRow<T, feature_vector,
             (n_out_cols + n_window_cols - 1) / stride> const& input,
    FilterTile<T, channel_vector, feature_vector, n_window_rows,
               n_window_cols> const& filter,
    OutputTile<T, channel_vector, n_out_rows, n_out_cols>& output,
    int const row_idx, int const first_col) {
  SNN_PRAGMA_UNROLL
  for (int out_row = 0; out_row < n_out_rows; ++out_row) {
    int const filter_row = row_idx - out_row;
    if (filter_row >= 0 && filter_row < n_window_rows) {
      convolve_1xw_one_row_bkin<stride>(input, filter, output, out_row,
                                        filter_row, first_col);
    }
  }
}
template <typename T, ConvType CType, int tile_rows, int tile_cols,
          int channel_vector_width, int feature_vector_width, bool use_fast_div,
          int window_rows, int window_cols, int static_stride = 0>
struct Conv2DTiledSYCL {
  using Index = int;
  using buffer_data = uint8_t;
  using index_div_type =
      typename fast_div::index_div<Index, use_fast_div>::type;
  static constexpr auto input_tile_width = tile_cols + window_cols - 1;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;
  Conv2DTiledSYCL(SYCLConv2DParams const& params, read_accessor const input,
                  read_accessor const kernel, write_accessor output) {}
  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {}
};
/**
 * Forward convolution using a tiled direct computation technique.
 *
 * This kernel can be vectorised in either the channels or the features. Both
 * significantly increases the number of registers required by the kernel, so
 * is unlikely to provide any additional performance. The feature vectorisation
 * can be controlled using the feature_vector_width template. The channel
 * vectorisation needs the kernel to be modified so that the loop over the
 * channels is split into a vectorised part and a scalar part.
 */
template <typename T, int tile_rows, int tile_cols, int channel_vector_width,
          int feature_vector_width, bool use_fast_div, int window_rows,
          int window_cols, int static_stride>
struct Conv2DTiledSYCL<T, ConvType::Forward, tile_rows, tile_cols,
                       channel_vector_width, feature_vector_width, use_fast_div,
                       window_rows, window_cols, static_stride> {
  using Index = int;
  using buffer_data = uint8_t;
  using index_div_type =
      typename fast_div::index_div<Index, use_fast_div>::type;
  static constexpr auto input_tile_width =
      (tile_cols - 1) * static_stride + window_cols;
  static constexpr auto CType = ConvType::Forward;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;
  using feature_vector = cl::sycl::vec<T, feature_vector_width>;

  Conv2DTiledSYCL(SYCLConv2DParams const& params, read_accessor const input,
                  read_accessor const kernel, write_accessor output)
      : n_tile_cols_{RoundRatioUpAboveZero(params.out_cols_, tile_cols)},
        n_tile_rows_{RoundRatioUpAboveZero(params.out_rows_, tile_rows)},
        n_elems_{params.batch_ * n_tile_rows_ * n_tile_cols_ *
                 params.features_ / feature_vector_width},
        n_feature_vectors_{params.features_ / feature_vector_width},
        div_feature_vectors_{n_feature_vectors_},
        div_n_tile_cols_{n_tile_cols_},
        div_n_tile_rows_{n_tile_rows_},
        SNN_CONSTRUCT_CONV_PARAMS(params),
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    const Index range = item.get_range().get(0);

    for (; index < n_elems_; index += range) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const helpers::TensorIndex4D tensor_idx =
          helpers::unflatten4d<Index, use_fast_div>(
              index, div_n_tile_rows_, n_tile_rows_, div_n_tile_cols_,
              n_tile_cols_, div_feature_vectors_, n_feature_vectors_);
      const Index feature = tensor_idx.s3 * feature_vector_width;
      const Index col_idx = tensor_idx.s2 * tile_cols;
      const Index row_idx = tensor_idx.s1 * tile_rows;
      const Index batch = tensor_idx.s0;

      const Index cstart = col_idx * static_stride - SNN_PARAM(pad_cols_);
      const Index rstart = row_idx * static_stride - SNN_PARAM(pad_rows_);

      OutputTile<T, feature_vector_width, tile_rows, tile_cols> out_tile{};
      const T* input_data_n = input_data +
                              batch * SNN_PARAM(in_cols_) *
                                  SNN_PARAM(in_rows_) * SNN_PARAM(channels_);
      Index channel = 0;
      if (channel_vector_width > 1) {
        for (; channel + channel_vector_width - 1 < SNN_PARAM(channels_);
             channel += channel_vector_width) {
          FilterTile<T, channel_vector_width, feature_vector_width, window_rows,
                     window_cols>
              filter_tile{kernel_data, channel, SNN_PARAM(channels_), feature,
                          SNN_PARAM(features_)};
          SNN_PRAGMA_UNROLL
          for (Index r = rstart, i = 0;
               i < window_rows + (tile_rows - 1) * static_stride; ++r, ++i) {
            if (r >= 0 && r < SNN_PARAM(in_rows_)) {
              InputRow<T, channel_vector_width, input_tile_width> input_tile{
                  input_data_n,         r,
                  SNN_PARAM(in_rows_),  cstart,
                  SNN_PARAM(in_cols_),  channel,
                  SNN_PARAM(channels_), check_bounds_tag{}};
              convolve_1xw_whole_tile_fwd<static_stride>(
                  input_tile, filter_tile, out_tile, i);
            }
          }
        }
      }
      for (; channel < SNN_PARAM(channels_); ++channel) {
        FilterTile<T, 1, feature_vector_width, window_rows, window_cols>
            filter_tile{kernel_data, channel, SNN_PARAM(channels_), feature,
                        SNN_PARAM(features_)};
        SNN_PRAGMA_UNROLL
        for (Index r = rstart, i = 0;
             i < window_rows + (tile_rows - 1) * static_stride; ++r, ++i) {
          if (r >= 0 && r < SNN_PARAM(in_rows_)) {
            InputRow<T, 1, input_tile_width> input_tile{
                input_data_n,         r,
                SNN_PARAM(in_rows_),  cstart,
                SNN_PARAM(in_cols_),  channel,
                SNN_PARAM(channels_), check_bounds_tag{}};
            convolve_1xw_whole_tile_fwd<static_stride>(input_tile, filter_tile,
                                                       out_tile, i);
          }
        }
      }
      out_tile.write_out(output_data, batch, row_idx, SNN_PARAM(out_rows_),
                         col_idx, SNN_PARAM(out_cols_), feature,
                         SNN_PARAM(features_));
    }
  }

 private:
  const Index n_tile_cols_;
  const Index n_tile_rows_;
  const Index n_elems_;
  const Index n_feature_vectors_;
  const index_div_type div_feature_vectors_;
  const index_div_type div_n_tile_cols_;
  const index_div_type div_n_tile_rows_;
  SNN_INJECT_CONV_PARAMS;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int tile_rows, int tile_cols, int channel_vector_width,
          int feature_vector_width, bool use_fast_div, int window_rows,
          int window_cols, int static_stride>
struct Conv2DTiledSYCL<T, ConvType::InputBackprop, tile_rows, tile_cols,
                       channel_vector_width, feature_vector_width, use_fast_div,
                       window_rows, window_cols, static_stride> {
  using Index = int;
  using buffer_data = uint8_t;
  using index_div_type =
      typename fast_div::index_div<Index, use_fast_div>::type;
  static constexpr auto input_tile_width =
      (tile_cols + window_cols - 1) / static_stride;
  static constexpr auto CType = ConvType::InputBackprop;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  Conv2DTiledSYCL(SYCLConv2DParams const& params, read_accessor const input,
                  read_accessor const kernel, write_accessor output)
      : n_tile_cols_{RoundRatioUpAboveZero(params.in_cols_, tile_cols)},
        n_tile_rows_{RoundRatioUpAboveZero(params.in_rows_, tile_rows)},
        n_elems_{params.batch_ * n_tile_rows_ * n_tile_cols_ *
                 params.channels_ / channel_vector_width},
        div_channels_{params.channels_ / channel_vector_width},
        div_n_tile_cols_{n_tile_cols_},
        div_n_tile_rows_{n_tile_rows_},
        SNN_CONSTRUCT_CONV_PARAMS(params),
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    const Index range = item.get_range().get(0);

    for (; index < n_elems_; index += range) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const helpers::TensorIndex4D tensor_idx =
          helpers::unflatten4d<Index, use_fast_div>(
              index, div_n_tile_rows_, n_tile_rows_, div_n_tile_cols_,
              n_tile_cols_, div_channels_,
              SNN_PARAM(channels_) / channel_vector_width);
      const Index channel = tensor_idx.s3 * channel_vector_width;
      const Index col_idx = tensor_idx.s2 * tile_cols;
      const Index row_idx = tensor_idx.s1 * tile_rows;
      const Index batch = tensor_idx.s0;

      const Index c = col_idx - SNN_PARAM(pad_cols_);
      const Index cstart = RoundRatioUp(c, static_stride);
      const Index first_col = c % static_stride;

      const Index r = row_idx - SNN_PARAM(pad_rows_);
      const Index rstart = RoundRatioUp(r, static_stride);
      const Index first_row = (r < 0 ? -r : r) % static_stride;

      OutputTile<T, channel_vector_width, tile_rows, tile_cols> out_tile{};
      const T* input_data_n = input_data +
                              batch * SNN_PARAM(out_cols_) *
                                  SNN_PARAM(out_rows_) * SNN_PARAM(features_);
      Index feature = 0;
      if (feature_vector_width > 1) {
        for (; feature + feature_vector_width - 1 < SNN_PARAM(features_);
             feature += feature_vector_width) {
          FilterTile<T, channel_vector_width, feature_vector_width, window_rows,
                     window_cols>
              filter_tile{kernel_data,          channel,
                          SNN_PARAM(channels_), feature,
                          SNN_PARAM(features_), mirror_filter_tag{}};
          SNN_PRAGMA_UNROLL
          for (Index r = rstart, i = first_row; i < window_rows + tile_rows - 1;
               ++r, i += static_stride) {
            if (r >= 0 && r < SNN_PARAM(out_rows_)) {
              InputRow<T, feature_vector_width, input_tile_width> input_tile{
                  input_data_n,         r,
                  SNN_PARAM(out_rows_), cstart,
                  SNN_PARAM(out_cols_), feature,
                  SNN_PARAM(features_), check_bounds_tag{}};
              convolve_1xw_whole_tile_bkin<static_stride>(
                  input_tile, filter_tile, out_tile, i, first_col);
            }
          }
        }
      }
      for (; feature < SNN_PARAM(features_); ++feature) {
        FilterTile<T, channel_vector_width, 1, window_rows, window_cols>
            filter_tile{kernel_data,          channel,
                        SNN_PARAM(channels_), feature,
                        SNN_PARAM(features_), mirror_filter_tag{}};
        SNN_PRAGMA_UNROLL
        for (Index r = rstart, i = first_row; i < window_rows + tile_rows - 1;
             ++r, i += static_stride) {
          if (r >= 0 && r < SNN_PARAM(out_rows_)) {
            InputRow<T, 1, input_tile_width> input_tile{
                input_data_n,         r,
                SNN_PARAM(out_rows_), cstart,
                SNN_PARAM(out_cols_), feature,
                SNN_PARAM(features_), check_bounds_tag{}};
            convolve_1xw_whole_tile_bkin<static_stride>(input_tile, filter_tile,
                                                        out_tile, i, first_col);
          }
        }
      }
      out_tile.write_out(output_data, batch, row_idx, SNN_PARAM(in_rows_),
                         col_idx, SNN_PARAM(in_cols_), channel,
                         SNN_PARAM(channels_));
    }
  }

 private:
  const Index n_tile_cols_;
  const Index n_tile_rows_;
  const Index n_elems_;
  const index_div_type div_channels_;
  const index_div_type div_n_tile_cols_;
  const index_div_type div_n_tile_rows_;
  SNN_INJECT_CONV_PARAMS;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
#if 0
/*
 * The main difference between the two backprop kernels is the way strides are
 * handled. In the filter backprop the input is strided and the kernel is not
 * whereas in the input backprop this is the other way around.
 *
 * For the filter backprop we are convolving the input with the output as the
 * filter. This means that the static window sizes are actually the
 * params.out_rows_ and params.out_cols_ rather than the params.window_*.
 */
template <typename T, bool use_fast_div, int static_out, int static_stride>
struct Conv2DSYCL<T, ConvType::FilterBackprop, use_fast_div, static_out,
                  static_stride> {
  using Index = int;
  using buffer_data = uint8_t;
  using index_div_type =
      typename fast_div::index_div<Index, use_fast_div>::type;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline SNN_ALWAYS_INLINE Conv2DSYCL(Index n_elems,
                                               const SYCLConv2DParams& params,
                                               const read_accessor input,
                                               const read_accessor kernel,
                                               write_accessor output)
      : n_elems_{n_elems},
        div_features_{params.features_},
        div_channels_{params.channels_},
        div_out_cols_{params.out_cols_},
        SNN_CONSTRUCT_CONV_PARAMS(params),
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const Index index = item.get_id(0);
    if (index < n_elems_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const helpers::TensorIndex4D tensor_idx =
          helpers::unflatten4d<Index, use_fast_div>(
              index, div_out_cols_, SNN_STATIC_PARAM(out, cols_), div_channels_,
              SNN_PARAM(channels_), div_features_, SNN_PARAM(features_));
      const Index feature = tensor_idx.s3;
      const Index channel = tensor_idx.s2;
      const Index col_idx = tensor_idx.s1;
      const Index row_idx = tensor_idx.s0;

      const Index cstart = col_idx - SNN_PARAM(pad_cols_);
      const Index cend =
          cl::sycl::min(cstart + SNN_PARAM(window_cols_), SNN_PARAM(in_cols_));

      const Index rstart = row_idx - SNN_PARAM(pad_rows_);
      const Index rend =
          cl::sycl::min(rstart + SNN_PARAM(window_rows_), SNN_PARAM(in_rows_));

      const Index filter_rows = RoundRatioUpAboveZero(
          SNN_PARAM(window_rows_), SNN_STATIC_PARAM(stride, rows_));
      const Index filter_cols = RoundRatioUpAboveZero(
          SNN_PARAM(window_cols_), SNN_STATIC_PARAM(stride, cols_));

      T out_val = static_cast<T>(0);
      const T* input_data_n = input_data;
      for (Index b = 0; b < SNN_PARAM(batch_); b++) {
        for (Index r = rstart, i = 0; r < rend;
             ++i, r += SNN_STATIC_PARAM(stride, rows_)) {
          if (r >= 0) {
            for (Index c = cstart, j = 0; c < cend;
                 ++j, c += SNN_STATIC_PARAM(stride, cols_)) {
              if (c >= 0) {
                const Index idx =
                    (r * SNN_PARAM(in_cols_) + c) * SNN_PARAM(channels_) +
                    channel;
                const Index k_idx = ((b * filter_rows + i) * filter_cols + j) *
                                        SNN_PARAM(features_) +
                                    feature;
                out_val += input_data_n[idx] * kernel_data[k_idx];
              }
            }
          }
        }
        input_data_n +=
            SNN_PARAM(in_cols_) * SNN_PARAM(in_rows_) * SNN_PARAM(channels_);
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const index_div_type div_features_;
  const index_div_type div_channels_;
  const index_div_type div_out_cols_;
  SNN_INJECT_CONV_PARAMS;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
#endif
}  // namespace direct
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_DIRECT_SYCL_KERNELS_H_
