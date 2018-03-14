#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_KERNELS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_KERNELS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_ops_sycl_common.h"
#include "tensorflow/core/kernels/conv_ops_sycl_kernel_helpers.h"
#include "tensorflow/core/kernels/conv_ops_sycl_param_macros.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace im2col {

template <typename T, int vector_width, ConvType CType>
struct ExtractInputTiles;
/**
 * Have one thread per input entry. That thread is then responsible for writing
 * its one entry to each point in the intermediate tensor as required for the
 * contraction.
 */
template <typename T, int vector_width>
struct ExtractInputTiles<T, vector_width, ConvType::Forward> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;
  using VecType = cl::sycl::vec<T, vector_width>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE ExtractInputTiles(
      Index const in_offset, Index const tile_size,
      SYCLConv2DParams const& params, read_accessor const input,
      write_accessor output)
      : n_items_{params.batch_ * params.in_rows_ * params.in_cols_ *
                 params.channels_ / vector_width},
        in_offset_{in_offset},
        tile_size_{tile_size},
        SNN_CONSTRUCT_CONV_PARAMS(params),
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_items_) {
      const T* input_data =
          ConvertToActualTypeSycl(T, input_accessor_) + in_offset_;
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);
      const Index in_idx = index * vector_width;
      VecType in_val = helpers::io::Load<VecType>()(input_data, in_idx);

      const helpers::TensorIndex4D tensor_idx =
          helpers::unflatten4d<Index, false>(
              index, SNN_PARAM(in_rows_), SNN_PARAM(in_rows_),
              SNN_PARAM(in_cols_), SNN_PARAM(in_cols_),
              SNN_PARAM(channels_) / vector_width,
              SNN_PARAM(channels_) / vector_width);
      const Index channel = tensor_idx.s3 * vector_width;
      const Index col_idx = tensor_idx.s2;
      const Index row_idx = tensor_idx.s1;
      const Index batch = tensor_idx.s0;

      // c is the index in the padded output tensor (ie with lots of extra
      // zeros), but without the first padding. first_padded_c adds this extra
      // padding.
      const Index c = col_idx + SNN_PARAM(pad_cols_);
      const Index first_padded_c =
          c - (SNN_PARAM(window_cols_) - 1) * SNN_PARAM(dilation_cols_);
      // The first and last output indices affected by this input.
      const Index last_used_c = c / SNN_PARAM(stride_cols_);
      Index cstart = RoundRatioUp(first_padded_c, SNN_PARAM(stride_cols_));
      const Index firstc = cstart * SNN_PARAM(stride_cols_) - first_padded_c;
      const Index cend = cl::sycl::min(last_used_c + 1, SNN_PARAM(out_cols_));

      const Index r = row_idx + SNN_PARAM(pad_rows_);
      const Index last_used_r = r / SNN_PARAM(stride_rows_);
      const Index first_padded_r =
          r - (SNN_PARAM(window_rows_) - 1) * SNN_PARAM(dilation_rows_);
      Index rstart = RoundRatioUp(first_padded_r, SNN_PARAM(stride_rows_));
      const Index firstr = rstart * SNN_PARAM(stride_rows_) - first_padded_r;
      const Index rend = cl::sycl::min(last_used_r + 1, SNN_PARAM(out_rows_));

      for (Index r = rstart, in_r = SNN_PARAM(window_rows_) - 1 - firstr;
           r < rend; ++r, in_r -= SNN_PARAM(stride_rows_)) {
        if (r >= 0) {
          for (Index c = cstart, in_c = SNN_PARAM(window_cols_) - 1 - firstc;
               c < cend; ++c, in_c -= SNN_PARAM(stride_cols_)) {
            if (c >= 0) {
              T* tile_start =
                  output_data +
                  ((batch * SNN_PARAM(out_rows_) + r) * SNN_PARAM(out_cols_) +
                   c) *
                      tile_size_;
              Index tile_idx = (in_r * SNN_PARAM(window_cols_) + in_c) *
                                   SNN_PARAM(channels_) +
                               channel;
              helpers::io::Store<VecType>()(tile_start, tile_idx, in_val);
            }
          }
        }
      }
    }
  }

 private:
  const Index n_items_;
  const Index in_offset_;
  const Index tile_size_;
  SNN_INJECT_CONV_PARAMS;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int vector_width>
struct ExtractInputTiles<T, vector_width, ConvType::InputBackprop> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;
  using VecType = cl::sycl::vec<T, vector_width>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE ExtractInputTiles(
      Index const in_offset, Index const tile_size,
      SYCLConv2DParams const& params, read_accessor const input,
      write_accessor output)
      : n_items_{params.batch_ * params.out_rows_ * params.out_cols_ *
                 params.features_ / vector_width},
        in_offset_{in_offset},
        tile_size_{tile_size},
        SNN_CONSTRUCT_CONV_PARAMS(params),
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_items_) {
      T const* const input_data =
          ConvertToActualTypeSycl(T, input_accessor_) + in_offset_;
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);
      const Index in_idx = index * vector_width;
      VecType in_val = helpers::io::Load<VecType>()(input_data, in_idx);

      const helpers::TensorIndex4D tensor_idx =
          helpers::unflatten4d<Index, false>(
              index, SNN_PARAM(out_rows_), SNN_PARAM(out_rows_),
              SNN_PARAM(out_cols_), SNN_PARAM(out_cols_),
              SNN_PARAM(features_) / vector_width,
              SNN_PARAM(features_) / vector_width);
      const Index feature = tensor_idx.s3 * vector_width;
      const Index col_idx = tensor_idx.s2;
      const Index row_idx = tensor_idx.s1;
      const Index batch = tensor_idx.s0;

      const Index cstart =
          col_idx * SNN_PARAM(stride_cols_) - SNN_PARAM(pad_cols_);
      const Index cend = cl::sycl::min(
          cstart + (SNN_PARAM(window_cols_) * SNN_PARAM(dilation_cols_)),
          SNN_PARAM(in_cols_));
      const Index firstc = cstart < 0 ? -cstart : 0;

      const Index rstart =
          row_idx * SNN_PARAM(stride_rows_) - SNN_PARAM(pad_rows_);
      const Index rend = cl::sycl::min(
          rstart + (SNN_PARAM(window_rows_) * SNN_PARAM(dilation_rows_)),
          SNN_PARAM(in_rows_));
      const Index firstr = rstart < 0 ? -rstart : 0;

      for (Index r = cl::sycl::max(rstart, 0),
                 in_r = SNN_PARAM(window_rows_) - 1 - firstr;
           r < rend; ++r, --in_r) {
        for (Index c = cl::sycl::max(cstart, 0),
                   in_c = SNN_PARAM(window_cols_) - 1 - firstc;
             c < cend; ++c, --in_c) {
          T* tile_start =
              output_data +
              ((batch * SNN_PARAM(in_rows_) + r) * SNN_PARAM(in_cols_) + c) *
                  tile_size_;
          Index tile_idx =
              (in_r * SNN_PARAM(window_cols_) + in_c) * SNN_PARAM(features_) +
              feature;
          helpers::io::Store<VecType>()(tile_start, tile_idx, in_val);
        }
      }
    }
  }

 private:
  const Index n_items_;
  const Index in_offset_;
  const Index tile_size_;
  SNN_INJECT_CONV_PARAMS;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int vector_width>
struct ExtractInputTiles<T, vector_width, ConvType::FilterBackprop> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE ExtractInputTiles(
      Index const in_offset, Index const tile_size,
      SYCLConv2DParams const& params, read_accessor const input,
      write_accessor output)
      : n_items_{params.batch_ * params.in_rows_ * params.in_cols_ *
                 params.channels_},
        in_offset_{in_offset},
        tile_size_{tile_size},
        SNN_CONSTRUCT_CONV_PARAMS(params),
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_items_) {
      const T* input_data =
          ConvertToActualTypeSycl(T, input_accessor_) + in_offset_;
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);
      T in_val = input_data[index];

      const helpers::TensorIndex4D tensor_idx =
          helpers::unflatten4d<Index, false>(
              index, SNN_PARAM(in_rows_), SNN_PARAM(in_rows_),
              SNN_PARAM(in_cols_), SNN_PARAM(in_cols_), SNN_PARAM(channels_),
              SNN_PARAM(channels_));
      const Index channel = tensor_idx.s3;
      const Index col_idx = tensor_idx.s2;
      const Index row_idx = tensor_idx.s1;
      const Index batch = tensor_idx.s0;

      // c is the index in the padded output tensor (ie with lots of extra
      // zeros), but without the first padding. first_padded_c adds this extra
      // padding.
      const Index c = col_idx + SNN_PARAM(pad_cols_);
      const Index first_padded_c =
          c - (SNN_PARAM(window_cols_) - 1) * SNN_PARAM(dilation_cols_);
      // The first and last output indices affected by this input.
      const Index last_used_c = c / SNN_PARAM(stride_cols_);
      const Index cstart =
          RoundRatioUp(first_padded_c, SNN_PARAM(stride_cols_));
      const Index cend = cl::sycl::min(last_used_c + 1, SNN_PARAM(out_cols_));

      const Index r = row_idx + SNN_PARAM(pad_rows_);
      const Index last_used_r = r / SNN_PARAM(stride_rows_);
      const Index first_padded_r =
          r - (SNN_PARAM(window_rows_) - 1) * SNN_PARAM(dilation_rows_);
      const Index rstart =
          RoundRatioUp(first_padded_r, SNN_PARAM(stride_rows_));
      const Index rend = cl::sycl::min(last_used_r + 1, SNN_PARAM(out_rows_));

      Index init_r = rstart;
      Index init_r_idx = SNN_PARAM(window_rows_) - 1;
      if (init_r < 0) {
        Index n_inc = RoundRatioUpAboveZero(-init_r, SNN_PARAM(dilation_rows_));
        init_r_idx -= n_inc * SNN_PARAM(stride_rows_);
        init_r += n_inc * SNN_PARAM(dilation_rows_);
      }
      Index init_c = cstart;
      Index init_c_idx = SNN_PARAM(window_cols_) - 1;
      if (init_c < 0) {
        Index n_inc = RoundRatioUpAboveZero(-init_c, SNN_PARAM(dilation_cols_));
        init_c_idx -= n_inc * SNN_PARAM(stride_cols_);
        init_c += n_inc * SNN_PARAM(dilation_cols_);
      }

      for (Index r = init_r, in_r = init_r_idx; r < rend;
           r += SNN_PARAM(dilation_rows_), in_r -= SNN_PARAM(stride_rows_)) {
        for (Index c = init_c, in_c = init_c_idx; c < cend;
             c += SNN_PARAM(dilation_cols_), in_c -= SNN_PARAM(stride_cols_)) {
          T* tile_start =
              output_data +
              ((r * SNN_PARAM(out_cols_) + c) * SNN_PARAM(channels_) +
               channel) *
                  tile_size_;
          Index tile_idx = ((batch * SNN_PARAM(window_rows_) + in_r) *
                                SNN_PARAM(window_cols_) +
                            in_c);
          tile_start[tile_idx] = in_val;
        }
      }
    }
  }

 private:
  const Index n_items_;
  const Index in_offset_;
  const Index tile_size_;
  SNN_INJECT_CONV_PARAMS;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T, ConvType CType>
struct ExtractKernelTiles;
template <typename T>
struct ExtractKernelTiles<T, ConvType::InputBackprop> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE ExtractKernelTiles(
      Index const in_offset, SYCLConv2DParams const& params,
      read_accessor const input, write_accessor output)
      : n_items_{params.window_rows_ * params.window_cols_ * params.channels_ *
                 params.features_},
        in_offset_{in_offset},
        n_window_rows_{params.window_rows_},
        n_window_cols_{params.window_cols_},
        n_channels_{params.channels_},
        n_features_{params.features_},
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_items_) {
      T const* const input_data =
          ConvertToActualTypeSycl(T, input_accessor_) + in_offset_;
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);
      T in_val = input_data[index];

      const helpers::TensorIndex4D tensor_idx =
          helpers::unflatten4d<Index, false>(
              index, n_window_cols_, n_window_cols_, n_channels_, n_channels_,
              n_features_, n_features_);
      const Index feature = tensor_idx.s3;
      const Index channel = tensor_idx.s2;
      const Index col = tensor_idx.s1;
      const Index row = tensor_idx.s0;

      const Index out_row = n_window_rows_ - 1 - row;
      const Index out_col = n_window_cols_ - 1 - col;
      const Index out_idx =
          ((out_row * n_window_cols_ + out_col) * n_features_ + feature) *
              n_channels_ +
          channel;
      output_data[out_idx] = in_val;
    }
  }

 private:
  const Index n_items_;
  const Index in_offset_;
  const Index n_window_rows_;
  const Index n_window_cols_;
  const Index n_channels_;
  const Index n_features_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
}  // namespace im2col
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_KERNELS_H_
