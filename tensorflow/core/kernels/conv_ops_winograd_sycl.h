/*
 * Copyright 2017 John Lawson, Codeplay Software.
 * All rights reserved.
 */

#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_H_

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
namespace functor {

template <typename T, int M, int N, int R, int S>
struct InputTile {
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  /**
   * Read the input data from the provided input array. The pointer is assumed
   * to be at the first value that should be read into the input tile.
   *
   * The input is expected to be in the NHWC data format.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename _T>
  inline TF_ATTRIBUTE_ALWAYS_INLINE InputTile(_T* input, int n_cols,
                                              int n_channels, int rsize,
                                              int csize, int firstr,
                                              int firstc) {
    for (int r = 0; r < A; ++r) {
      for (int c = 0; c < B; ++c) {
        data[r][c] = (r < firstr || c < firstc || r >= rsize || c >= csize)
                         ? static_cast<_T>(0)
                         : input[(r * n_cols + c) * n_channels];
      }
    }
  }
  T data[A][B];
};
template <typename T, int M, int N, int R, int S>
struct BaseFilterTile {
  T data[R][S];
};
template <typename T, int M, int N, int R, int S, ConvType CType>
struct FilterTile;
template <typename T, int M, int N, int R, int S>
struct FilterTile<T, M, N, R, S, ConvType::Forward> final
    : public BaseFilterTile<T, M, N, R, S> {
  using BaseFilterTile<T, M, N, R, S>::data;
  /**
   * Read the filter data from the provided input array. The pointer is assumed
   * to be at the first value that should be read into the input tile.
   *
   * The input is expected to be in (Height x Width x Channel x Feature) format.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename _T>
  inline TF_ATTRIBUTE_ALWAYS_INLINE FilterTile(_T* input, int channel,
                                               int feature, int n_cols,
                                               int n_channels, int n_features) {
    for (int r = 0; r < R; ++r) {
      for (int c = 0; c < S; ++c) {
        int idx =
            ((r * n_cols + c) * n_channels + channel) * n_features + feature;
        data[r][c] = input[idx];
      }
    }
  }
};
template <typename T, int M, int N, int R, int S>
struct FilterTile<T, M, N, R, S, ConvType::InputBackprop> final
    : public BaseFilterTile<T, M, N, R, S> {
  using BaseFilterTile<T, M, N, R, S>::data;
  /**
   * Read the filter data from the provided input array but mirror the filter
   * for use in backprop. The pointer is assumed to be at the first value that
   * should be read into the input tile.
   *
   * The input is expected to be in (Height x Width x Channel x Feature) format.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename _T>
  inline TF_ATTRIBUTE_ALWAYS_INLINE FilterTile(
      _T const* input, int const channel, int const feature, int const n_cols,
      int const n_channels, int const n_features) {
    for (int r = 0; r < R; ++r) {
      for (int c = 0; c < S; ++c) {
        // Here the transforms (R - 1 - r) and (S - 1 - c) mirror the filter
        // data. Note that the channel and feature dims were switched in the
        // kernel params.
        int idx =
            (((R - 1 - r) * n_cols + (S - 1 - c)) * n_features + feature) *
                n_channels +
            channel;
        data[r][c] = input[idx];
      }
    }
  }
};
template <typename T, int M, int N, int R, int S>
struct FilterTile<T, M, N, R, S, ConvType::FilterBackprop> final
    : public BaseFilterTile<T, M, N, R, S> {
  using BaseFilterTile<T, M, N, R, S>::data;
  /**
   * Read the filter data from the provided input array.
   *
   * The input is expected to be in (Batch x Height x Width x Feature) format.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename _T>
  inline TF_ATTRIBUTE_ALWAYS_INLINE FilterTile(_T const* input,
                                               SYCLOutputWindow const& w,
                                               int const n_cols,
                                               int const n_features) {
    input += w.offset;
    for (int r = 0; r < R; ++r) {
      for (int c = 0; c < S; ++c) {
        int idx = (r * n_cols + c) * n_features;
        data[r][c] =
            (r >= w.rsize || c >= w.csize) ? static_cast<_T>(0) : input[idx];
      }
    }
  }
};
/**
 * Base class for the output tile which provides a correctly sized data array.
 */
template <typename T, int M, int N, int R, int S>
struct BaseTransformedFilterTile {
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  T data[A][B];
};
/**
 * Tile which transforms the intermediate layer into the output layer. Should be
 * specialised for each Winograd tranform.
 *
 * This object needs to provide the following constructor:
 *
 *   template <bool mirror>
 *   TransformedFilterTile(FilterTile<T, 2, 2, 3, 3, mirror> const& filter)
 *       : BaseTransformedFilterTile<T, 2, 2, 3, 3>{} {
 *     // Implement the filter Winograd transform
 *   }
 *
 * and needs to provide access to the base class data array with:
 *
 *   using BaseTransformedFilterTile<T, 2, 2, 3, 3>::data;
 */
template <typename T, int M, int N, int R, int S>
struct TransformedFilterTile;
/**
 * Base class for the output tile which provides a correctly sized data array.
 */
template <typename T, int M, int N, int R, int S>
struct BaseTransformedInputTile {
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  T data[A][B];
};
/**
 * Tile which transforms the intermediate layer into the output layer. Should be
 * specialised for each Winograd tranform.
 *
 * This object needs to provide the following constructor:
 *
 *   TransformedInputTile(InputTile<T, 2, 2, 3, 3> const& inp)
 *       : BaseTransformedInputTile<T, 2, 2, 3, 3>{} {
 *     // Implement the input Winograd transform
 *   }
 *
 * and needs to provide access to the base class data array with:
 *
 *   using BaseTransformedInpuTile<T, 2, 2, 3, 3>::data;
 */
template <typename T, int M, int N, int R, int S>
struct TransformedInputTile;
/**
 * Tile to store the intermediate Winograd data. Provides an update method to
 * increment the tile with provided transformed inputs and filters.
 */
template <typename T, int M, int N, int R, int S>
struct IntermediateTile {
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  inline TF_ATTRIBUTE_ALWAYS_INLINE IntermediateTile() {
    for (int i = 0; i < A; ++i) {
      for (int j = 0; j < B; ++j) {
        data[i][j] = static_cast<T>(0);
      }
    }
  }
  /**
   * Read the intermediate tile from a temporary buffer. The input shape is
   * expected to be
   *   [ (M+R-1)(N+S-1), features, (batch * tile_rows * tile_cols) ].
   */
  template <typename _T>
  inline TF_ATTRIBUTE_ALWAYS_INLINE IntermediateTile(_T* input,
                                                     int const tile_idx,
                                                     int const n_tiles,
                                                     int const feature,
                                                     int const n_features) {
    input += feature * n_tiles + tile_idx;
    for (int r = 0; r < A; ++r) {
      for (int c = 0; c < B; ++c) {
        data[r][c] = *input;
        input += n_features * n_tiles;
      }
    }
  }
  inline TF_ATTRIBUTE_ALWAYS_INLINE void update(
      TransformedInputTile<T, M, N, R, S> const& inp,
      TransformedFilterTile<T, M, N, R, S> const& filter) {
    for (int r = 0; r < A; ++r) {
      for (int c = 0; c < B; ++c) {
        data[r][c] += inp.data[r][c] * filter.data[r][c];
      }
    }
  }
  T data[A][B];
};
/**
 * Base class for the output tile which provides a correctly sized data array.
 */
template <typename T, int M, int N, int R, int S>
struct BaseOutputTile {
  T data[M][N];
};
/**
 * Tile which transforms the intermediate layer into the output layer. Should be
 * specialised for each Winograd tranform.
 *
 * This object needs to provide the following constructor:
 *
 *   OutputTile(IntermediateTile<T, M, N, R, S> const& tile)
 *       : BaseOutputTile<T, M, N, R, S>{} {
 *     // Implement the inverse Winograd transform
 *   }
 *
 * and needs to provide access to the base class data array with:
 *
 *   using BaseOutputTile<T, M, N, R, S>::data;
 */
template <typename T, int M, int N, int R, int S>
struct OutputTile;
template <typename T, int M, int N, int R, int S>
struct OutputData {
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  using Index = int;
  /**
   * Write the transformed input tile to a temporary buffer where each entry of
   * the tile is split into separate matrices. The output pointer should be at
   * the start of the temporary buffer.
   *
   * The resulting temporary buffer will be written as a batch of these
   * matrices, with a shape of
   *   [ (M+R-1)*(N+S-1), (batch * row_tiles * col_tiles), channels ].
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename _T>
  inline TF_ATTRIBUTE_ALWAYS_INLINE static void write_transformed_input(
      _T* output, Index const tile_idx, Index const channel,
      Index const n_tiles, Index const n_channels,
      TransformedInputTile<T, M, N, R, S> const& tile) {
    output += tile_idx * n_channels + channel;
    for (int r = 0; r < A; ++r) {
      for (int c = 0; c < B; ++c) {
        *output = tile.data[r][c];
        output += n_tiles * n_channels;
      }
    }
  }
  /**
   * Write the transformed filter tile to a temporary buffer where each entry of
   * the tile is split into separate matrices. The output pointer should be at
   * the start of the temporary buffer.
   *
   * The resulting temporary buffer will be written as a batch of these
   * matrices, with a shape of
   *   [ (M+R-1)*(N+S-1), features, channels ].
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename _T>
  inline TF_ATTRIBUTE_ALWAYS_INLINE static void write_transformed_filter(
      _T* output, Index const channel, Index const feature,
      Index const n_channels, Index const n_features,
      TransformedFilterTile<T, M, N, R, S> const& tile) {
    output += feature * n_channels + channel;
    for (int r = 0; r < A; ++r) {
      for (int c = 0; c < B; ++c) {
        *output = tile.data[r][c];
        output += n_features * n_channels;
      }
    }
  }
  /**
   * Write the output tile to the correct output memory. The output pointer
   * should be at the start of the output buffer. The resulting output shape is
   * NHWC.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename _T>
  inline TF_ATTRIBUTE_ALWAYS_INLINE static void write_output(
      _T* output, SYCLOutputWindow const& window, Index const n_cols,
      Index const n_channels, OutputTile<T, M, N, R, S> const& tile) {
    output += window.offset;
    for (int r = 0; r < M && r < window.rsize; ++r) {
      for (int c = 0; c < N && c < window.csize; ++c) {
        output[(r * n_cols + c) * n_channels] = tile.data[r][c];
      }
    }
  }
  /**
   * Write the output tile to the correct output memory. The output pointer
   * should be at the start of the output buffer. The resulting output shape is
   * NHWC.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename _T>
  inline TF_ATTRIBUTE_ALWAYS_INLINE static void write_filter_output(
      _T* output, Index const channel, Index const feature,
      Index const n_channels, Index const n_features,
      OutputTile<T, M, N, R, S> const& tile) {
    output += channel * n_features + feature;
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++c) {
        *output = tile.data[r][c];
        output += n_channels * n_features;
      }
    }
  }
};
/**
 * Kernel function object to compute the forward pass of a convolution using a
 * simple Winograd transform. This kernel will perform all the computation in
 * the convolution.
 */
template <typename T, int M, int N, int R, int S, ConvType CType>
struct WinogradConv {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE WinogradConv(Index const n_tiles,
                                                 SYCLConv2DParams const& params,
                                                 read_accessor const input,
                                                 read_accessor const kernel,
                                                 write_accessor output)
      : n_tiles_{n_tiles},
        p_{params},
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get(0);
    if (index < n_tiles_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index feature = index % p_.features_;
      const Index tile_idx = index / p_.features_;
      const SYCL2DWindow w = p_.winograd_input_window<M, N, R, S>(
          tile_idx, CType == ConvType::InputBackprop);

      IntermediateTile<T, M, N, R, S> tmp{};

      for (Index channel = 0; channel < p_.channels_; ++channel) {
        const Index offset =
            ((w.batch * p_.in_rows_ + w.rstart) * p_.in_cols_ + w.cstart) *
                p_.channels_ +
            channel;
        InputTile<T, M, N, R, S> inp(input_data + offset, p_.in_cols_,
                                     p_.channels_, w.rend - w.rstart,
                                     w.cend - w.cstart, w.firstr, w.firstc);
        FilterTile<T, M, N, R, S, CType> filter(kernel_data, channel, feature,
                                                p_.window_cols_, p_.channels_,
                                                p_.features_);
        tmp.update(TransformedInputTile<T, M, N, R, S>{inp},
                   TransformedFilterTile<T, M, N, R, S>{filter});
      }
      SYCLOutputWindow out_w = p_.winograd_output_index<M, N, R, S>(index);
      OutputData<T, M, N, R, S>::write_output(output_data, out_w, p_.out_cols_,
                                              p_.features_,
                                              OutputTile<T, M, N, R, S>{tmp});
    }
  }

 private:
  const Index n_tiles_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int M, int N, int R, int S, ConvType CType>
struct ExtractInputTiles {
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
      Index const n_threads, Index const n_tiles,
      SYCLConv2DParams const& params, read_accessor const input,
      write_accessor output)
      : n_threads_{n_threads},
        n_tiles_{n_tiles},
        p_{params},
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get(0);
    if (index < n_threads_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index n_tile_rows = RoundRatioUpAboveZero(p_.out_rows_, M);
      const Index n_tile_cols = RoundRatioUpAboveZero(p_.out_cols_, N);
      const Index channel_idx = index % p_.channels_;
      const Index tile_idx = index / p_.channels_;

      Index batch = tile_idx;
      const Index cstart = (batch % n_tile_cols) * N - p_.pad_cols_;
      const Index cend = std::min(cstart + N + S - 1, p_.in_cols_);
      const Index firstc = cstart < 0 ? -cstart : 0;
      batch /= n_tile_cols;

      const Index rstart = (batch % n_tile_rows) * M - p_.pad_rows_;
      const Index rend = std::min(rstart + M + R - 1, p_.in_rows_);
      const Index firstr = rstart < 0 ? -rstart : 0;
      batch /= n_tile_rows;

      const Index offset =
          ((batch * p_.in_rows_ + rstart) * p_.in_cols_ + cstart) *
              p_.channels_ +
          channel_idx;
      InputTile<T, M, N, R, S> inp(input_data + offset, p_.in_cols_,
                                   p_.channels_, rend - rstart, cend - cstart,
                                   firstr, firstc);

      OutputData<T, M, N, R, S>::write_transformed_input(
          output_data, index / p_.channels_, channel_idx, n_tiles_,
          p_.channels_, TransformedInputTile<T, M, N, R, S>{inp});
    }
  }

 private:
  const Index n_threads_;
  const Index n_tiles_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int M, int N, int R, int S>
struct ExtractInputTiles<T, M, N, R, S, ConvType::FilterBackprop> {
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
      Index const n_threads, Index const n_tile_rows, Index const n_tile_cols,
      SYCLConv2DParams const& params, read_accessor const input,
      write_accessor output)
      : n_threads_{n_threads},
        n_tile_rows_{n_tile_rows},
        n_tile_cols_{n_tile_cols},
        p_{params},
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get(0);
    if (index < n_threads_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index channel_idx = index % p_.channels_;
      const Index tile_idx = index / p_.channels_;

      Index batch = tile_idx;
      const Index cstart = (batch % n_tile_cols_) * S - p_.pad_cols_;
      const Index cend = std::min(cstart + N + S - 1, p_.in_cols_);
      const Index firstc = cstart < 0 ? -cstart : 0;
      batch /= n_tile_cols_;

      const Index rstart = (batch % n_tile_rows_) * R - p_.pad_rows_;
      const Index rend = std::min(rstart + M + R - 1, p_.in_rows_);
      const Index firstr = rstart < 0 ? -rstart : 0;
      batch /= n_tile_rows_;

      const Index offset =
          ((batch * p_.in_rows_ + rstart) * p_.in_cols_ + cstart) *
              p_.channels_ +
          channel_idx;
      InputTile<T, M, N, R, S> inp(input_data + offset, p_.in_cols_,
                                   p_.channels_, rend - rstart, cend - cstart,
                                   firstr, firstc);
      TransformedInputTile<T, M, N, R, S> trans{inp};

      const Index n_tiles = p_.batch_ * n_tile_rows_ * n_tile_cols_;
      OutputData<T, M, N, R, S>::write_transformed_input(
          output_data, tile_idx, channel_idx, n_tiles, p_.channels_, trans);
    }
  }

 private:
  const Index n_threads_;
  const Index n_tile_rows_;
  const Index n_tile_cols_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int M, int N, int R, int S, ConvType CType>
struct ExtractKernelTiles {
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
      Index const n_threads, Index const /*n_tiles*/,
      SYCLConv2DParams const& params, read_accessor const kernel,
      write_accessor output)
      : n_threads_{n_threads},
        p_{params},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get(0);
    if (index < n_threads_) {
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index feature_idx = index % p_.features_;
      const Index channel_idx = (index / p_.features_) % p_.channels_;

      FilterTile<T, M, N, R, S, CType> filter(kernel_data, channel_idx,
                                              feature_idx, p_.window_cols_,
                                              p_.channels_, p_.features_);
      TransformedFilterTile<T, M, N, R, S> transformed{filter};

      OutputData<T, M, N, R, S>::write_transformed_filter(
          output_data, channel_idx, feature_idx, p_.channels_, p_.features_,
          transformed);
    }
  }

 private:
  const Index n_threads_;
  const SYCLConv2DParams p_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int M, int N, int R, int S>
struct ExtractKernelTiles<T, M, N, R, S, ConvType::FilterBackprop> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto CType = ConvType::FilterBackprop;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE ExtractKernelTiles(
      Index const n_threads, Index const n_tiles,
      SYCLConv2DParams const& params, read_accessor const kernel,
      write_accessor output)
      : n_threads_{n_threads},
        n_tiles_{n_tiles},
        p_{params},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get(0);
    if (index < n_threads_) {
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index feature = index % p_.features_;
      const Index tile_idx = index / p_.features_;

      SYCLOutputWindow w =
          p_.winograd_kernel_from_tile<M, N, R, S>(tile_idx, feature);

      FilterTile<T, M, N, R, S, CType> filter(kernel_data, w, p_.window_cols_,
                                              p_.features_);
      TransformedFilterTile<T, M, N, R, S> transformed{filter};

      OutputData<T, M, N, R, S>::write_transformed_filter(
          output_data, feature, tile_idx, p_.features_, n_tiles_, transformed);
    }
  }

 private:
  const Index n_threads_;
  const Index n_tiles_;
  const SYCLConv2DParams p_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int M, int N, int R, int S, ConvType CType>
struct ExtractOutputTiles {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE ExtractOutputTiles(
      Index const n_threads, Index const n_tiles,
      SYCLConv2DParams const& params, read_accessor const input,
      write_accessor output)
      : n_threads_{n_threads},
        n_tiles_{n_tiles},
        p_{params},
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get(0);
    if (index < n_threads_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index tile_idx = index % n_tiles_;
      const Index feature = index / n_tiles_;

      IntermediateTile<T, M, N, R, S> tmp{input_data, tile_idx, n_tiles_,
                                          feature, p_.features_};
      SYCLOutputWindow out_w =
          p_.winograd_output_index<M, N, R, S>(tile_idx, feature);
      OutputData<T, M, N, R, S>::write_output(output_data, out_w, p_.out_cols_,
                                              p_.features_,
                                              OutputTile<T, M, N, R, S>{tmp});
    }
  }

 private:
  const Index n_threads_;
  const Index n_tiles_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int M, int N, int R, int S>
struct ExtractOutputTiles<T, M, N, R, S, ConvType::FilterBackprop> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE ExtractOutputTiles(
      Index const n_threads, Index const /*n_tiles*/,
      SYCLConv2DParams const& params, read_accessor const input,
      write_accessor output)
      : n_threads_{n_threads},
        p_{params},
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get(0);
    if (index < n_threads_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index feature = index % p_.features_;
      const Index channel = index / p_.features_;

      IntermediateTile<T, M, N, R, S> tmp{input_data, feature, p_.features_,
                                          channel, p_.channels_};
      OutputData<T, M, N, R, S>::write_filter_output(
          output_data, channel, feature, p_.channels_, p_.features_,
          OutputTile<T, M, N, R, S>{tmp});
    }
  }

 private:
  const Index n_threads_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int C1 = 1, int C2 = 1>
struct BatchMatmul {
  using TensorType = Eigen::Tensor<T, 3, Eigen::RowMajor, Eigen::DenseIndex>;
  using Tensor = Eigen::TensorMap<TensorType, Eigen::Aligned>;
  using ConstTensorType =
      Eigen::Tensor<T const, 3, Eigen::RowMajor, Eigen::DenseIndex>;
  using ConstTensor = Eigen::TensorMap<ConstTensorType, Eigen::Aligned>;
  using ContractDims = Eigen::IndexPairList<Eigen::type2indexpair<C1, C2>>;

  void operator()(Eigen::SyclDevice const& d, ConstTensor const& in_x,
                  ConstTensor const& in_y, Tensor& out) {
    auto const& dims_x = in_x.dimensions();
    int const batches = dims_x[0];

    for (int i = 0; i < batches; ++i) {
      auto x = in_x.template chip<0>(i);
      auto y = in_y.template chip<0>(i);
      auto z = out.template chip<0>(i);
      z.device(d) = x.contract(y, ContractDims{});
    }
  }
};
}
template <typename T, int M, int N, int R, int S, ConvType CType>
struct LaunchMatmulWinograd {
  using Index = int;
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;

  static T* launch_input_transform(Eigen::SyclDevice const& device,
                                   const Tensor& tensor_in, Index const n_tiles,
                                   const SYCLConv2DParams& params) {
    using Functor = functor::ExtractInputTiles<T, M, N, R, S, CType>;
    static constexpr auto read_mode = Functor::read_mode;
    static constexpr auto write_mode = Functor::write_mode;

    size_t const transform_size =
        A * B * n_tiles * params.channels_ * sizeof(T);
    T* const transform = static_cast<T*>(device.allocate(transform_size));
    T const* const input = tensor_in.template flat<T>().data();

    const Index workgroup_size = device.maxSyclThreadsPerBlock();
    const Index n_items = n_tiles * params.channels_;
    const Index n_threads = RoundUpToNearestMultiple(n_items, workgroup_size);

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access = device.get_sycl_accessor<read_mode>(cgh, input);
      auto transform_access =
          device.get_sycl_accessor<write_mode>(cgh, transform);

      Functor extract_fun(n_items, n_tiles, params, input_access,
                          transform_access);
      cgh.parallel_for(cl::sycl::range<1>(n_threads), extract_fun);
    });
    return transform;
  }
  static T* launch_kernel_transform(Eigen::SyclDevice const& device,
                                    const Tensor& filter, Index const n_tiles,
                                    const SYCLConv2DParams& params) {
    using Functor = functor::ExtractKernelTiles<T, M, N, R, S, CType>;
    static constexpr auto read_mode = Functor::read_mode;
    static constexpr auto write_mode = Functor::write_mode;

    size_t const transform_size =
        A * B * params.channels_ * params.features_ * sizeof(T);
    T* const transform = static_cast<T*>(device.allocate(transform_size));
    T const* const input = filter.template flat<T>().data();

    const Index workgroup_size = device.maxSyclThreadsPerBlock();
    const Index n_items = params.features_ * params.channels_;
    const Index n_threads = RoundUpToNearestMultiple(n_items, workgroup_size);

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access = device.get_sycl_accessor<read_mode>(cgh, input);
      auto transform_access =
          device.get_sycl_accessor<write_mode>(cgh, transform);

      Functor extract_fun(n_items, n_tiles, params, input_access,
                          transform_access);
      cgh.parallel_for(cl::sycl::range<1>(n_threads), extract_fun);
    });
    return transform;
  }
  static T* launch_batch_matmul(Eigen::SyclDevice const& device,
                                T const* const input, T const* const filter,
                                Index const n_tiles, Index const n_channels,
                                Index const n_features) {
    using ConstTensorType =
        Eigen::Tensor<T const, 3, Eigen::RowMajor, Eigen::DenseIndex>;
    using ConstTensor = Eigen::TensorMap<ConstTensorType, Eigen::Aligned>;
    using TensorType = Eigen::Tensor<T, 3, Eigen::RowMajor, Eigen::DenseIndex>;
    using Tensor = Eigen::TensorMap<TensorType, Eigen::Aligned>;
    using TensorShape = Eigen::DSizes<Eigen::DenseIndex, 3>;

    TensorShape const in_shape{A * B, n_tiles, n_channels};
    ConstTensor in_tensor{input, in_shape};
    TensorShape const fil_shape{A * B, n_features, n_channels};
    ConstTensor fil_tensor{filter, fil_shape};

    size_t const n_out_bytes = A * B * n_tiles * n_features * sizeof(T);
    T* const output = static_cast<T*>(device.allocate(n_out_bytes));
    TensorShape const out_shape{A * B, n_features, n_tiles};
    Tensor out_tensor{output, out_shape};

    functor::BatchMatmul<T>()(device, fil_tensor, in_tensor, out_tensor);
    return output;
  }
  static void launch_output_transform(Eigen::SyclDevice const& device,
                                      T const* const input, Index const n_tiles,
                                      SYCLConv2DParams const& params,
                                      Tensor* const out) {
    using Functor = functor::ExtractOutputTiles<T, M, N, R, S, CType>;
    static constexpr auto read_mode = Functor::read_mode;
    static constexpr auto write_mode = Functor::write_mode;

    T* const output = out->template flat<T>().data();

    const Index workgroup_size = device.maxSyclThreadsPerBlock();
    const Index n_items = n_tiles * params.features_;
    const Index n_threads = RoundUpToNearestMultiple(n_items, workgroup_size);

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access = device.get_sycl_accessor<read_mode>(cgh, input);
      auto out_access = device.get_sycl_accessor<write_mode>(cgh, output);

      Functor extract_fun(n_items, n_tiles, params, input_access, out_access);
      cgh.parallel_for(cl::sycl::range<1>(n_threads), extract_fun);
    });
  }
  static bool launch(OpKernelContext* context, Tensor* output,
                     const Tensor& tensor_in, const Tensor& filter,
                     const SYCLConv2DParams& params) {
    auto const& device = context->eigen_device<Eigen::SyclDevice>();
    const Index n_out_tile_rows = RoundRatioUpAboveZero(params.out_rows_, M);
    const Index n_out_tile_cols = RoundRatioUpAboveZero(params.out_cols_, N);
    const Index n_out_tiles = params.batch_ * n_out_tile_rows * n_out_tile_cols;

    T* const in_transform =
        launch_input_transform(device, tensor_in, n_out_tiles, params);
    T* const fil_transform =
        launch_kernel_transform(device, filter, n_out_tiles, params);
    T* const intermediate =
        launch_batch_matmul(device, in_transform, fil_transform, n_out_tiles,
                            params.channels_, params.features_);
    launch_output_transform(device, intermediate, n_out_tiles, params, output);

    device.deallocate(intermediate);
    device.deallocate(fil_transform);
    device.deallocate(in_transform);
    return true;
  }
};
template <typename T, int M, int N, int R, int S>
struct LaunchMatmulWinograd<T, M, N, R, S, ConvType::FilterBackprop> {
  using Index = int;
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  static constexpr auto CType = ConvType::FilterBackprop;

  static T* launch_input_transform(Eigen::SyclDevice const& device,
                                   const Tensor& tensor_in,
                                   Index const n_tile_rows,
                                   Index const n_tile_cols,
                                   const SYCLConv2DParams& params) {
    using Functor = functor::ExtractInputTiles<T, M, N, R, S, CType>;
    static constexpr auto read_mode = Functor::read_mode;
    static constexpr auto write_mode = Functor::write_mode;

    const Index n_tiles = params.batch_ * n_tile_rows * n_tile_cols;
    size_t const transform_size =
        A * B * n_tiles * params.channels_ * sizeof(T);
    T* const transform = static_cast<T*>(device.allocate(transform_size));
    T const* const input = tensor_in.template flat<T>().data();

    const Index workgroup_size = device.maxSyclThreadsPerBlock();
    const Index n_items = n_tiles * params.channels_;
    const Index n_threads = RoundUpToNearestMultiple(n_items, workgroup_size);

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access = device.get_sycl_accessor<read_mode>(cgh, input);
      auto transform_access =
          device.get_sycl_accessor<write_mode>(cgh, transform);

      Functor extract_fun(n_items, n_tile_rows, n_tile_cols, params,
                          input_access, transform_access);
      cgh.parallel_for(cl::sycl::range<1>(n_threads), extract_fun);
    });
    return transform;
  }
  static T* launch_kernel_transform(Eigen::SyclDevice const& device,
                                    const Tensor& filter, Index const n_tiles,
                                    const SYCLConv2DParams& params) {
    using Functor = functor::ExtractKernelTiles<T, M, N, R, S, CType>;
    static constexpr auto read_mode = Functor::read_mode;
    static constexpr auto write_mode = Functor::write_mode;

    size_t const transform_size =
        A * B * n_tiles * params.features_ * sizeof(T);
    T* const transform = static_cast<T*>(device.allocate(transform_size));
    T const* const input = filter.template flat<T>().data();

    const Index workgroup_size = device.maxSyclThreadsPerBlock();
    const Index n_items = params.features_ * n_tiles;
    const Index n_threads = RoundUpToNearestMultiple(n_items, workgroup_size);

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access = device.get_sycl_accessor<read_mode>(cgh, input);
      auto transform_access =
          device.get_sycl_accessor<write_mode>(cgh, transform);

      Functor extract_fun(n_items, n_tiles, params, input_access,
                          transform_access);
      cgh.parallel_for(cl::sycl::range<1>(n_threads), extract_fun);
    });
    return transform;
  }
  static T* launch_batch_matmul(Eigen::SyclDevice const& device,
                                T const* const input, T const* const filter,
                                Index const n_tiles, Index const n_channels,
                                Index const n_features) {
    using ConstTensorType =
        Eigen::Tensor<T const, 3, Eigen::RowMajor, Eigen::DenseIndex>;
    using ConstTensor = Eigen::TensorMap<ConstTensorType, Eigen::Aligned>;
    using TensorType = Eigen::Tensor<T, 3, Eigen::RowMajor, Eigen::DenseIndex>;
    using Tensor = Eigen::TensorMap<TensorType, Eigen::Aligned>;
    using TensorShape = Eigen::DSizes<Eigen::DenseIndex, 3>;

    TensorShape const in_shape{A * B, n_tiles, n_channels};
    ConstTensor in_tensor{input, in_shape};
    TensorShape const fil_shape{A * B, n_tiles, n_features};
    ConstTensor fil_tensor{filter, fil_shape};

    size_t const n_out_bytes = A * B * n_channels * n_features * sizeof(T);
    T* const output = static_cast<T*>(device.allocate(n_out_bytes));
    TensorShape const out_shape{A * B, n_channels, n_features};
    Tensor out_tensor{output, out_shape};

    functor::BatchMatmul<T, 0, 0>()(device, in_tensor, fil_tensor, out_tensor);
    return output;
  }
  static void launch_output_transform(Eigen::SyclDevice const& device,
                                      T const* const input, Index const n_tiles,
                                      SYCLConv2DParams const& params,
                                      Tensor* const out) {
    using Functor = functor::ExtractOutputTiles<T, M, N, R, S, CType>;
    static constexpr auto read_mode = Functor::read_mode;
    static constexpr auto write_mode = Functor::write_mode;

    T* const output = out->template flat<T>().data();

    const Index workgroup_size = device.maxSyclThreadsPerBlock();
    const Index n_items = params.channels_ * params.features_;
    const Index n_threads = RoundUpToNearestMultiple(n_items, workgroup_size);

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access = device.get_sycl_accessor<read_mode>(cgh, input);
      auto out_access = device.get_sycl_accessor<write_mode>(cgh, output);

      Functor extract_fun(n_items, n_tiles, params, input_access, out_access);
      cgh.parallel_for(cl::sycl::range<1>(n_threads), extract_fun);
    });
  }
  static bool launch(OpKernelContext* context, Tensor* output,
                     const Tensor& tensor_in, const Tensor& filter,
                     const SYCLConv2DParams& params) {
    auto const& device = context->eigen_device<Eigen::SyclDevice>();
    const Index n_tile_rows = RoundRatioUpAboveZero(params.window_rows_, R);
    const Index n_tile_cols = RoundRatioUpAboveZero(params.window_cols_, S);
    const Index n_tiles = params.batch_ * n_tile_rows * n_tile_cols;

    T* const in_transform = launch_input_transform(
        device, tensor_in, n_tile_rows, n_tile_cols, params);
    T* const fil_transform =
        launch_kernel_transform(device, filter, n_tiles, params);
    T* const intermediate =
        launch_batch_matmul(device, in_transform, fil_transform, n_tiles,
                            params.channels_, params.features_);
    launch_output_transform(device, intermediate, n_tiles, params, output);

    device.deallocate(intermediate);
    device.deallocate(fil_transform);
    device.deallocate(in_transform);
    return true;
  }
};
}
#include "tensorflow/core/kernels/conv_ops_winograd_sycl_impl.h"

#endif  // TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_H_
