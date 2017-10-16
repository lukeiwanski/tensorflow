/*
 * Copyright 2017 John Lawson, Codeplay Software.
 * All rights reserved.
 */
#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_SYCL_COMMON_H_
#define TENSORFLOW_KERNELS_CONV_OPS_SYCL_COMMON_H_

namespace tensorflow {
/**
 * Helper function to provide the ratio of two integers, always rounded up. If
 * the numerator is negative then we assume that the rounded ratio will be
 * zero, otherwise we need to ensure that the value is rounded up rather
 * than down.
 */
template <typename IntegerType>
inline TF_ATTRIBUTE_ALWAYS_INLINE IntegerType
RoundRatioUpAboveZero(const IntegerType num, const IntegerType div) {
  static_assert(std::is_integral<IntegerType>::value,
                "RoundRatioUpAboveZero only valid for integral types");
  return num < 0 ? 0 : (num + div - 1) / div;
}
/**
 * Helper function to round up an integral value to the nearest multiple of a
 * given multiplier.
 */
template <typename IntegerType>
inline TF_ATTRIBUTE_ALWAYS_INLINE IntegerType
RoundUpToNearestMultiple(IntegerType val, const IntegerType multiplier) {
  static_assert(std::is_integral<IntegerType>::value,
                "RoundUpToNearestMultiple only valid for integral types");
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
        pad_cols_{static_cast<Index>(pad_cols)} {}

  const Index channels_;
  const Index features_;
  const Index batch_;

  const Index in_rows_;
  const Index in_cols_;

  const Index window_rows_;
  const Index window_cols_;

  const Index stride_rows_;
  const Index stride_cols_;

  const Index out_rows_;
  const Index out_cols_;

  const Index pad_rows_;
  const Index pad_cols_;

  /**
   * Get the index in the kernel tensor for a particular channel, row and
   * column.
   */
  inline TF_ATTRIBUTE_ALWAYS_INLINE Index kernel_index(const Index channel,
                                                       const Index feature,
                                                       const Index i,
                                                       const Index j) const {
    return (((i * window_cols_) + j) * channels_ + channel) * features_ +
           feature;
  }
  /**
   * Get the index in the kernel tensor for the kernel backprop for a
   * particular channel, row and column. Here we have to mirror the kernel
   * indices to match how the backprop is computed.
   */
  inline TF_ATTRIBUTE_ALWAYS_INLINE Index backprop_index(const Index channel,
                                                         const Index feature,
                                                         const Index i,
                                                         const Index j) const {
    const Index mirrored_row = window_rows_ - i - 1;
    const Index mirrored_col = window_cols_ - j - 1;
    return ((mirrored_row * window_cols_ + mirrored_col) * features_ +
            channel) *
               channels_ +
           feature;
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
    return ((batch * filter_rows + i) * filter_cols + j) * features_ + feature;
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
    const Index cend = std::min(cstart + window_cols_, in_cols_);
    const Index firstc = cstart < 0 ? -cstart : 0;
    batch /= out_cols_;

    const Index rstart = (batch % out_rows_) * stride_rows_ - pad_rows_;
    const Index rend = std::min(rstart + window_rows_, in_rows_);
    const Index firstr = rstart < 0 ? -rstart : 0;
    batch /= out_rows_;

    return {rstart, rend, firstr, cstart, cend, firstc, batch};
  }
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCL2DWindow
  output_window_from_input(const Index tile_idx) const {
    static_assert(std::is_integral<Index>::value,
                  "Index must be an integral type");
    static_assert(std::is_signed<Index>::value, "Index must be a signed type");
    Index n = tile_idx;
    // c is the index in the padded output tensor (ie with lots of extra zeros),
    // but without the first padding. first_padded_c adds this extra padding.
    const Index c = (n % in_cols_) + pad_cols_;
    const Index first_padded_c = c - window_cols_ + 1;
    // The first and last output indices affected by this input.
    const Index last_used_c = c / stride_cols_;
    const Index first_used_c =
        RoundRatioUpAboveZero(first_padded_c, stride_cols_);

    const Index offset_c = first_used_c * stride_cols_ - first_padded_c;
    const Index cend = std::min(last_used_c + 1, out_cols_);
    n /= in_cols_;

    const Index r = (n % in_rows_) + pad_rows_;
    const Index last_used_r = r / stride_rows_;
    const Index first_padded_r = r - window_rows_ + 1;
    const Index first_used_r =
        RoundRatioUpAboveZero(first_padded_r, stride_rows_);

    const Index offset_r = first_used_r * stride_rows_ - first_padded_r;
    const Index rend = std::min(last_used_r + 1, out_rows_);
    n /= in_rows_;

//    printf("ii: %i, r: %i, rs: %i, re: %i, ofr: %i, c: %i, cs: %i, ce: %i, ofc: %i, batch: %i\n",
//        tile_idx, r, first_used_r, rend, offset_r, c, first_used_c, cend, offset_c, n);
    return {first_used_r, rend, offset_r, first_used_c, cend, offset_c, n};
  }
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCL2DKernelWindow
  kernel_window_from_output(const Index index) const {
    static_assert(std::is_integral<Index>::value,
                  "Index must be an integral type");
    static_assert(std::is_signed<Index>::value, "Index must be a signed type");
    Index n = index;
    const Index feature = n % features_;
    n /= features_;
    const Index channel = n % channels_;
    n /= channels_;

    Index cstart = n % out_cols_ - pad_cols_;
    const Index cend = std::min(cstart + window_cols_, in_cols_);
    const Index firstc = cstart < 0 ? -cstart : 0;
    n /= out_cols_;

    Index rstart = n - pad_rows_;
    const Index rend = std::min(rstart + window_rows_, in_rows_);
    const Index firstr = rstart < 0 ? -rstart : 0;

    return {rstart, rend, firstr, cstart, cend, firstc, feature, channel};
  }
  template <int M, int N, int R, int S>
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCL2DWindow
  winograd_input_window(const Index index, bool backprop) const {
    static_assert(std::is_integral<Index>::value,
                  "Index must be an integral type");
    static_assert(std::is_signed<Index>::value, "Index must be a signed type");
    if (!backprop) {
      const Index no_out_tile_rows = RoundRatioUpAboveZero(out_rows_, M);
      const Index no_out_tile_cols = RoundRatioUpAboveZero(out_cols_, N);
      Index batch = index;

      const Index cstart = (batch % no_out_tile_cols) * N - pad_cols_;
      const Index cend = std::min(cstart + N + S - 1, in_cols_);
      const Index firstc = cstart < 0 ? -cstart : 0;
      batch /= no_out_tile_cols;

      const Index rstart = (batch % no_out_tile_rows) * M - pad_rows_;
      const Index rend = std::min(rstart + M + R - 1, in_rows_);
      const Index firstr = rstart < 0 ? -rstart : 0;
      batch /= no_out_tile_rows;

      return {rstart, rend, firstr, cstart, cend, firstc, batch};
    } else {
      const Index no_out_tile_rows = RoundRatioUpAboveZero(out_rows_, M);
      const Index no_out_tile_cols = RoundRatioUpAboveZero(out_cols_, N);
      Index n = index;

      const Index cstart = (n % no_out_tile_cols) * N + pad_cols_ - S + 1;
      const Index firstc = cstart < 0 ? -cstart : 0;
      const Index cend = std::min(cstart + N + S - 1, in_cols_);
      n /= no_out_tile_cols;

      const Index rstart = (n % no_out_tile_rows) * M + pad_rows_ - R + 1;
      const Index firstr = rstart < 0 ? -rstart : 0;
      const Index rend = std::min(rstart + M + R - 1, in_rows_);
      n /= no_out_tile_rows;

      return {rstart, rend, firstr, cstart, cend, firstc, n};
    }
  }
  template <int M, int N, int R, int S>
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCLOutputWindow
  winograd_output_index(const Index tile_idx, const Index feature) const {
    const Index no_out_tile_rows = RoundRatioUpAboveZero(out_rows_, M);
    const Index no_out_tile_cols = RoundRatioUpAboveZero(out_cols_, N);
    Index batch = tile_idx;

    const Index col = batch % no_out_tile_cols * N;
    const Index cend = std::min(col + N, out_cols_);
    batch /= no_out_tile_cols;

    const Index row = batch % no_out_tile_rows * M;
    const Index rend = std::min(row + M, out_rows_);
    batch /= no_out_tile_rows;

    const Index offset =
        ((batch * out_rows_ + row) * out_cols_ + col) * features_ + feature;
    return {rend - row, cend - col, offset};
  }
  template <int M, int N, int R, int S>
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCLOutputWindow
  winograd_kernel_from_tile(const Index tile_idx, const Index feature) const {
    const Index n_tile_rows = RoundRatioUpAboveZero(window_rows_, R);
    const Index n_tile_cols = RoundRatioUpAboveZero(window_cols_, S);
    Index batch = tile_idx;

    const Index col = batch % n_tile_cols * S;
    const Index cend = std::min(col + N, window_cols_);
    batch /= n_tile_cols;

    const Index row = batch % n_tile_rows * R;
    const Index rend = std::min(row + M, window_rows_);
    batch /= n_tile_rows;

    const Index offset =
        ((batch * window_rows_ + row) * window_cols_ + col) * features_ +
        feature;
    return {rend - row, cend - col, offset};
  }
};
}
#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_COMMON_H_
