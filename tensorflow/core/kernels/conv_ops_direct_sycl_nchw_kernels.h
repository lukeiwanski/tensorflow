#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_DIRECT_SYCL_NCHW_KERNELS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_DIRECT_SYCL_NCHW_KERNELS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_ops_sycl_common.h"
#include "tensorflow/core/kernels/conv_ops_sycl_fast_div.h"
#include "tensorflow/core/kernels/conv_ops_sycl_kernel_helpers.h"
#include "tensorflow/core/kernels/conv_ops_sycl_kernel_macros.h"
#include "tensorflow/core/kernels/conv_ops_sycl_param_macros.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
namespace direct {
template <typename T, ConvType CType, bool use_fast_div = false,
          int static_window = 0, int static_stride = 0>
struct Conv2DNCHW;
template <typename T, bool use_fast_div, int static_window, int static_stride>
struct Conv2DNCHW<T, ConvType::Forward, use_fast_div, static_window,
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

  inline Conv2DNCHW(Index n_elems, const SYCLConv2DParams& params,
                    const read_accessor input, const read_accessor kernel,
                    write_accessor output)
      : n_elems_{params.batch_ * params.features_ * params.out_rows_ *
                 params.out_cols_},
        div_features_{params.features_},
        div_out_cols_{params.out_cols_},
        div_out_rows_{params.out_rows_},
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
              index, div_features_, SNN_PARAM(features_), div_out_rows_,
              SNN_PARAM(out_rows_), div_out_cols_, SNN_PARAM(out_cols_));
      const Index col_idx = tensor_idx.s3;
      const Index row_idx = tensor_idx.s2;
      const Index feature = tensor_idx.s1;
      const Index batch = tensor_idx.s0;

      const Index cstart =
          col_idx * SNN_STATIC_PARAM(stride, cols_) - SNN_PARAM(pad_cols_);
      const Index rstart =
          row_idx * SNN_STATIC_PARAM(stride, rows_) - SNN_PARAM(pad_rows_);

      T out_val = static_cast<T>(0);
      const T* input_data_n = input_data +
                              batch * SNN_PARAM(in_cols_) *
                                  SNN_PARAM(in_rows_) * SNN_PARAM(channels_);
      for (Index r = rstart, i = 0;
           i < SNN_STATIC_PARAM(window, rows_) && r < SNN_PARAM(in_rows_);
           ++r, ++i) {
        if (r >= 0) {
          for (Index c = cstart, j = 0;
               j < SNN_STATIC_PARAM(window, cols_) && c < SNN_PARAM(in_cols_);
               ++c, ++j) {
            if (c >= 0) {
              for (Index channel = 0; channel < SNN_PARAM(channels_);
                   ++channel) {
                const Index idx =
                    (channel * SNN_PARAM(in_rows_) + r) * SNN_PARAM(in_cols_) +
                    c;
                const Index k_idx = ((i * SNN_STATIC_PARAM(window, cols_) + j) *
                                         SNN_PARAM(channels_) +
                                     channel) *
                                        SNN_PARAM(features_) +
                                    feature;
                out_val += input_data_n[idx] * kernel_data[k_idx];
              }
            }
          }
        }
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const index_div_type div_features_;
  const index_div_type div_out_cols_;
  const index_div_type div_out_rows_;
  SNN_INJECT_CONV_PARAMS;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
template <typename T, bool use_fast_div, int static_window, int static_stride>
struct Conv2DNCHW<T, ConvType::InputBackprop, use_fast_div, static_window,
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

  inline Conv2DNCHW(Index n_elems, const SYCLConv2DParams& params,
                    const read_accessor input, const read_accessor kernel,
                    write_accessor output)
      : n_elems_{params.batch_ * params.features_ * params.in_rows_ *
                 params.in_cols_},
        div_features_{params.features_},
        div_in_rows_{params.in_rows_},
        div_in_cols_{params.in_cols_},
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
              index, div_features_, SNN_PARAM(features_), div_in_rows_,
              SNN_PARAM(in_rows_), div_in_cols_, SNN_PARAM(in_cols_));
      const Index col_idx = tensor_idx.s3;
      const Index row_idx = tensor_idx.s2;
      const Index feature = tensor_idx.s1;
      const Index batch = tensor_idx.s0;

      // c is the index in the padded output tensor (ie with lots of extra
      // zeros), but without the first padding. first_padded_c adds this extra
      // padding.
      const Index c = col_idx + SNN_PARAM(pad_cols_);
      const Index first_padded_c = c - SNN_STATIC_PARAM(window, cols_) + 1;
      // The first and last output indices affected by this input.
      const Index last_used_c = c / SNN_STATIC_PARAM(stride, cols_);
      const Index first_used_c = RoundRatioUpAboveZero(
          first_padded_c, SNN_STATIC_PARAM(stride, cols_));

      const Index firstc =
          first_used_c * SNN_STATIC_PARAM(stride, cols_) - first_padded_c;
      const Index cstart = cl::sycl::max(first_used_c, static_cast<Index>(0));
      const Index cend = cl::sycl::min(last_used_c + 1, SNN_PARAM(out_cols_));

      const Index r = row_idx + SNN_PARAM(pad_rows_);
      const Index last_used_r = r / SNN_STATIC_PARAM(stride, rows_);
      const Index first_padded_r = r - SNN_STATIC_PARAM(window, rows_) + 1;
      const Index first_used_r = RoundRatioUpAboveZero(
          first_padded_r, SNN_STATIC_PARAM(stride, rows_));

      const Index firstr =
          first_used_r * SNN_STATIC_PARAM(stride, rows_) - first_padded_r;
      const Index rstart = cl::sycl::max(first_used_r, static_cast<Index>(0));
      const Index rend = cl::sycl::min(last_used_r + 1, SNN_PARAM(out_rows_));

      T out_val = static_cast<T>(0);
      const T* input_data_n = input_data +
                              batch * SNN_PARAM(out_cols_) *
                                  SNN_PARAM(out_rows_) * SNN_PARAM(channels_);
      for (Index r = rstart, i = firstr; r < rend;
           ++r, i += SNN_STATIC_PARAM(stride, rows_)) {
        for (Index c = cstart, j = firstc; c < cend;
             ++c, j += SNN_STATIC_PARAM(stride, cols_)) {
          for (Index channel = 0; channel < SNN_PARAM(channels_); ++channel) {
            const Index idx =
                (channel * SNN_PARAM(out_rows_) + r) * SNN_PARAM(out_cols_) + c;

            const Index mirrored_row = SNN_STATIC_PARAM(window, rows_) - i - 1;
            const Index mirrored_col = SNN_STATIC_PARAM(window, cols_) - j - 1;
            const Index k_idx =
                ((mirrored_row * SNN_STATIC_PARAM(window, cols_) +
                  mirrored_col) *
                     SNN_PARAM(features_) +
                 feature) *
                    SNN_PARAM(channels_) +
                channel;
            out_val += input_data_n[idx] * kernel_data[k_idx];
          }
        }
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const index_div_type div_features_;
  const index_div_type div_in_rows_;
  const index_div_type div_in_cols_;
  SNN_INJECT_CONV_PARAMS;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
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
struct Conv2DNCHW<T, ConvType::FilterBackprop, use_fast_div, static_out,
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

  inline Conv2DNCHW(Index n_elems, const SYCLConv2DParams& params,
                    const read_accessor input, const read_accessor kernel,
                    write_accessor output)
      : n_elems_{params.out_rows_ * params.out_cols_ * params.channels_ *
                 params.features_},
        div_features_{params.features_},
        div_channels_{params.channels_},
        div_out_cols_{params.out_cols_},
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
                    (channel * SNN_PARAM(in_rows_) + r) * SNN_PARAM(in_cols_) +
                    c;

                const Index k_idx =
                    ((b * SNN_PARAM(features_) + feature) * filter_rows + i) *
                        filter_cols +
                    j;
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
}  // namespace direct
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_DIRECT_SYCL_NCHW_KERNELS_H_
