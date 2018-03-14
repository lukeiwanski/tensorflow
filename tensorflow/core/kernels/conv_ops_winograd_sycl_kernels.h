#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_KERNELS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_KERNELS_H_

#include "tensorflow/core/kernels/conv_ops_sycl_kernel_helpers.h"

namespace tensorflow {
namespace winograd {

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
  template <typename _T, typename Index>
  inline SNN_ALWAYS_INLINE InputTile(_T* input, Index const batch,
                                     Index const rstart, Index const n_rows,
                                     Index const cstart, Index const n_cols,
                                     Index const channel,
                                     Index const n_channels, Index const firstr,
                                     Index const firstc) {
    Index const offset =
        ((batch * n_rows + rstart) * n_cols + cstart) * n_channels + channel;
    input += offset;
    SNN_PRAGMA_UNROLL
    for (int r = 0; r < A; ++r) {
      SNN_PRAGMA_UNROLL
      for (int c = 0; c < B; ++c) {
        data[r][c] =
            (r < firstr || c < firstc || r + rstart >= n_rows ||
             c + cstart >= n_cols)
                ? static_cast<T>(0)
                : helpers::io::Load<T>()(input, (r * n_cols + c) * n_channels);
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
   * to be at the start of the filter tensor.
   *
   * The input is expected to be in (Height x Width x Channel x Feature) format.
   * The height of the filter (no. of rows) is expected to be R, and the width
   * (no. of cols) is S.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename _T>
  inline SNN_ALWAYS_INLINE FilterTile(_T const* input, int const channel,
                                      int const feature, int const n_channels,
                                      int const n_features) {
    input += channel * n_features + feature;
    SNN_PRAGMA_UNROLL
    for (int r = 0; r < R; ++r) {
      SNN_PRAGMA_UNROLL
      for (int c = 0; c < S; ++c) {
        int idx = (r * S + c) * n_channels * n_features;
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
   * for use in backprop. The pointer is assumed to be at the start of the
   * filter tensor.
   *
   * The input is expected to be in (Height x Width x Channel x Feature) format.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename _T>
  inline SNN_ALWAYS_INLINE FilterTile(_T const* input, int const channel,
                                      int const feature, int const n_channels,
                                      int const n_features) {
    input += channel * n_features + feature;
    SNN_PRAGMA_UNROLL
    for (int r = 0; r < R; ++r) {
      SNN_PRAGMA_UNROLL
      for (int c = 0; c < S; ++c) {
        // Here the transforms (R - 1 - r) and (S - 1 - c) mirror the filter
        // data. Note that the channel and feature dims were switched in the
        // kernel params.
        int idx = (r * S + c) * n_channels * n_features;
        data[R - 1 - r][S - 1 - c] = input[idx];
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
  inline SNN_ALWAYS_INLINE FilterTile(_T const* input,
                                      SYCLOutputWindow const& w,
                                      int const n_cols, int const n_features) {
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
  /**
   * Read the intermediate tile from a temporary buffer. The input shape is
   * expected to be
   *   [ (M+R-1)(N+S-1), (batch * tile_rows * tile_cols), features ].
   */
  template <typename _T>
  inline SNN_ALWAYS_INLINE IntermediateTile(_T* input, int const tile_idx,
                                            int const n_tiles,
                                            int const feature,
                                            int const n_features) {
    input += tile_idx * n_features + feature;
    for (int r = 0; r < A; ++r) {
      for (int c = 0; c < B; ++c) {
        const int idx = (r * B + c) * n_features * n_tiles;
        data[r][c] = input[idx];
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
  inline SNN_ALWAYS_INLINE static void write_transformed_input(
      _T* output, Index const tile_idx, Index const channel,
      Index const n_tiles, Index const n_channels,
      TransformedInputTile<T, M, N, R, S> const& tile) {
    output += tile_idx * n_channels + channel;
    for (int r = 0; r < A; ++r) {
      for (int c = 0; c < B; ++c) {
        const int idx = (r * B + c) * n_tiles * n_channels;
        helpers::io::Store<T>()(output, idx, tile.data[r][c]);
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
  inline SNN_ALWAYS_INLINE static void write_transformed_filter(
      _T* output, Index const channel, Index const feature,
      Index const n_channels, Index const n_features,
      TransformedFilterTile<T, M, N, R, S> const& tile) {
    output += feature * n_channels + channel;
    for (int r = 0; r < A; ++r) {
      for (int c = 0; c < B; ++c) {
        const int idx = (r * B + c) * n_features * n_channels;
        output[idx] = tile.data[r][c];
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
  inline SNN_ALWAYS_INLINE static void write_output(
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
   * HWCF.
   *
   * The filter has size M x N when run in FilterBackprop mode, so we don't need
   * to check the bounds for writing to the output.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <bool accumulate_output, typename _T>
  inline SNN_ALWAYS_INLINE static void write_filter_output(
      _T* output, Index const channel, Index const feature,
      Index const n_channels, Index const n_features,
      OutputTile<T, M, N, R, S> const& tile) {
    output += channel * n_features + feature;
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++c) {
        const int idx = (r * N + c) * n_channels * n_features;
        if (accumulate_output) {
          output[idx] += tile.data[r][c];
        } else {
          output[idx] = tile.data[r][c];
        }
      }
    }
  }
};
template <typename T, int channel_vector, int M, int N, int R, int S,
          ConvType CType>
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
  using VecType = cl::sycl::vec<T, channel_vector>;

  ExtractInputTiles(Index const in_offset, Index const n_tiles,
                    Index const n_tile_rows, Index const n_tile_cols,
                    SYCLConv2DParams const& params, read_accessor const input,
                    write_accessor output)
      : in_offset_{in_offset},
        n_elems_{params.channels_ * n_tile_cols * n_tile_rows * params.batch_ /
                 channel_vector},
        n_tiles_{n_tiles},
        n_tile_rows_{n_tile_rows},
        n_tile_cols_{n_tile_cols},
        n_in_cols_{params.in_cols_},
        n_in_rows_{params.in_rows_},
        n_channels_{params.channels_},
        n_pad_cols_{params.pad_cols_},
        n_pad_rows_{params.pad_rows_},
        input_accessor_{input},
        output_accessor_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_elems_) {
      const T* input_data =
          ConvertToActualTypeSycl(T, input_accessor_) + in_offset_;
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const helpers::TensorIndex2D tile_channel_idx =
          helpers::unflatten2d<Index, false>(index,
                                             n_channels_ / channel_vector,
                                             n_channels_ / channel_vector);
      const Index channel_idx = tile_channel_idx.s1 * channel_vector;
      const Index tile_idx = tile_channel_idx.s0;

      const helpers::TensorIndex3D tile_tensor_idx =
          helpers::unflatten3d<Index, false>(
              tile_idx, n_tile_rows_, n_tile_rows_, n_tile_cols_, n_tile_cols_);
      const Index col_idx = tile_tensor_idx.s2;
      const Index row_idx = tile_tensor_idx.s1;
      const Index batch = tile_tensor_idx.s0;

      const Index cstart = col_idx * N - n_pad_cols_;
      const Index firstc = cstart < 0 ? -cstart : 0;

      const Index rstart = row_idx * M - n_pad_rows_;
      const Index firstr = rstart < 0 ? -rstart : 0;

      InputTile<VecType, M, N, R, S> inp(
          input_data, batch, rstart, n_in_rows_, cstart, n_in_cols_,
          channel_idx, n_channels_, firstr, firstc);

      OutputData<VecType, M, N, R, S>::write_transformed_input(
          output_data, tile_idx, channel_idx, n_tiles_, n_channels_,
          TransformedInputTile<VecType, M, N, R, S>{inp});
    }
  }

 private:
  const Index in_offset_;
  const Index n_elems_;
  const Index n_tiles_;
  const Index n_tile_rows_;
  const Index n_tile_cols_;
  const Index n_in_cols_;
  const Index n_in_rows_;
  const Index n_channels_;
  const Index n_pad_cols_;
  const Index n_pad_rows_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int channel_vector, int M, int N, int R, int S>
struct ExtractInputTiles<T, channel_vector, M, N, R, S, ConvType::FilterBackprop> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;
  using VecType = cl::sycl::vec<T, channel_vector>;

  ExtractInputTiles(Index const in_offset, Index const n_tiles,
                    Index const n_tile_rows, Index const n_tile_cols,
                    SYCLConv2DParams const& params, read_accessor const input,
                    write_accessor output)
      : in_offset_{in_offset},
        n_elems_{params.channels_ * n_tile_cols * n_tile_rows * params.batch_ /
                 channel_vector},
        n_tiles_{n_tiles},
        n_tile_rows_{n_tile_rows},
        n_tile_cols_{n_tile_cols},
        n_in_cols_{params.in_cols_},
        n_in_rows_{params.in_rows_},
        n_channels_{params.channels_},
        n_pad_cols_{params.pad_cols_},
        n_pad_rows_{params.pad_rows_},
        input_accessor_{input},
        output_accessor_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_elems_) {
      const T* input_data =
          ConvertToActualTypeSycl(T, input_accessor_) + in_offset_;
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const helpers::TensorIndex2D tile_channel_idx =
          helpers::unflatten2d<Index, false>(index,
                                             n_channels_ / channel_vector,
                                             n_channels_ / channel_vector);
      const Index channel_idx = tile_channel_idx.s1 * channel_vector;
      const Index tile_idx = tile_channel_idx.s0;

      const helpers::TensorIndex3D tile_tensor_idx =
          helpers::unflatten3d<Index, false>(
              tile_idx, n_tile_rows_, n_tile_rows_, n_tile_cols_, n_tile_cols_);
      const Index col_idx = tile_tensor_idx.s2;
      const Index row_idx = tile_tensor_idx.s1;
      const Index batch = tile_tensor_idx.s0;

      const Index cstart = col_idx * S - n_pad_cols_;
      const Index firstc = cstart < 0 ? -cstart : 0;

      const Index rstart = row_idx * R - n_pad_rows_;
      const Index firstr = rstart < 0 ? -rstart : 0;

      InputTile<VecType, M, N, R, S> inp(input_data, batch, rstart, n_in_rows_,
                                         cstart, n_in_cols_, channel_idx,
                                         n_channels_, firstr, firstc);
      TransformedInputTile<VecType, M, N, R, S> trans{inp};

      OutputData<VecType, M, N, R, S>::write_transformed_input(
          output_data, tile_idx, channel_idx, n_tiles_, n_channels_, trans);
    }
  }

 private:
  const Index in_offset_;
  const Index n_elems_;
  const Index n_tiles_;
  const Index n_tile_rows_;
  const Index n_tile_cols_;
  const Index n_in_cols_;
  const Index n_in_rows_;
  const Index n_channels_;
  const Index n_pad_cols_;
  const Index n_pad_rows_;
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

  ExtractKernelTiles(Index const /*n_tiles*/, SYCLConv2DParams const& params,
                     read_accessor const kernel, write_accessor output)
      : n_tiles_{params.channels_ * params.features_},
        n_channels_{params.channels_},
        n_features_{params.features_},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_tiles_) {
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const helpers::TensorIndex2D channel_feature_idx =
          helpers::unflatten2d<Index, false>(index, n_features_, n_features_);
      const Index feature_idx = channel_feature_idx.s1;
      const Index channel_idx = channel_feature_idx.s0;

      FilterTile<T, M, N, R, S, CType> filter(
          kernel_data, channel_idx, feature_idx, n_channels_, n_features_);
      TransformedFilterTile<T, M, N, R, S> transformed{filter};

      OutputData<T, M, N, R, S>::write_transformed_filter(
          output_data, channel_idx, feature_idx, n_channels_, n_features_,
          transformed);
    }
  }

 private:
  const Index n_tiles_;
  const Index n_channels_;
  const Index n_features_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int M, int N, int R, int S>
struct ExtractKernelTiles<T, M, N, R, S, ConvType::InputBackprop> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto CType = ConvType::InputBackprop;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  /*
   * Note that for the input backprop the features and channels in params have
   * been switched. params.channels_ are most packed in memory, which we expect
   * to be n_features_ in the kernel. We switch these back in the constructor
   * so they are as expected.
   */
  ExtractKernelTiles(Index const /*n_tiles*/, SYCLConv2DParams const& params,
                     read_accessor const kernel, write_accessor output)
      : n_tiles_{params.channels_ * params.features_},
        n_features_{params.channels_},
        n_channels_{params.features_},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_tiles_) {
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const helpers::TensorIndex2D feature_channel_idx =
          helpers::unflatten2d<Index, false>(index, n_features_, n_features_);
      const Index feature_idx = feature_channel_idx.s1;
      const Index channel_idx = feature_channel_idx.s0;

      FilterTile<T, M, N, R, S, CType> filter(
          kernel_data, channel_idx, feature_idx, n_channels_, n_features_);
      TransformedFilterTile<T, M, N, R, S> transformed{filter};
      /*
       * Here we can either write out with features or channels packed in
       * memory. The matmul will perform best if the features are packed, as
       * then the matmul will read this in a packed way, if instead the channels
       * are packed then the matmul will require a transpose which may affect
       * performance. The problem is that in this kernel we read the data with
       * channels packed, so writing out with features packed will require a
       * large number of strided writes.
       *
       * Here we write out with channels packed, and so the matmul must be
       * called with this tensor transposed.
       */
      OutputData<T, M, N, R, S>::write_transformed_filter(
          output_data, feature_idx, channel_idx, n_features_, n_channels_,
          transformed);
    }
  }

 private:
  const Index n_tiles_;
  const Index n_features_;
  const Index n_channels_;
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

  ExtractKernelTiles(Index const in_offset, Index const n_tiles,
                     Index const n_tile_rows, Index const n_tile_cols,
                     SYCLConv2DParams const& params, read_accessor const kernel,
                     write_accessor output)
      : in_offset_{in_offset},
        n_threads_{params.features_ * n_tile_cols * n_tile_rows *
                   params.batch_},
        n_tiles_{n_tiles},
        n_tile_rows_{n_tile_rows},
        n_tile_cols_{n_tile_cols},
        n_window_rows_{params.window_rows_},
        n_window_cols_{params.window_cols_},
        n_features_{params.features_},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_threads_) {
      const T* kernel_data =
          ConvertToActualTypeSycl(T, kernel_accessor_) + in_offset_;
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const helpers::TensorIndex2D tile_feature_idx =
          helpers::unflatten2d<Index, false>(index, n_features_, n_features_);
      const Index tile_idx = tile_feature_idx.s0;
      const Index feature = tile_feature_idx.s1;

      const helpers::TensorIndex3D tile_tensor_idx =
          helpers::unflatten3d<Index, false>(
              tile_idx, n_tile_rows_, n_tile_rows_, n_tile_cols_, n_tile_cols_);
      const Index col_idx = tile_tensor_idx.s2;
      const Index row_idx = tile_tensor_idx.s1;
      const Index batch = tile_tensor_idx.s0;

      const Index col = col_idx * S;
      const Index cend = cl::sycl::min(col + N, n_window_cols_);

      const Index row = row_idx * R;
      const Index rend = cl::sycl::min(row + M, n_window_rows_);

      const Index offset =
          ((batch * n_window_rows_ + row) * n_window_cols_ + col) *
              n_features_ +
          feature;
      SYCLOutputWindow w{rend - row, cend - col, offset};

      FilterTile<T, M, N, R, S, CType> filter(kernel_data, w, n_window_cols_,
                                              n_features_);
      TransformedFilterTile<T, M, N, R, S> transformed{filter};

      OutputData<T, M, N, R, S>::write_transformed_filter(
          output_data, feature, tile_idx, n_features_, n_tiles_, transformed);
    }
  }

 private:
  const Index in_offset_;
  const Index n_threads_;
  const Index n_tiles_;
  const Index n_tile_rows_;
  const Index n_tile_cols_;
  const Index n_window_rows_;
  const Index n_window_cols_;
  const Index n_features_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int M, int N, int R, int S, ConvType CType,
          bool accumulate_output = false>
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

  ExtractOutputTiles(Index const out_offset, Index const n_tiles,
                     Index const n_tile_rows, Index const n_tile_cols,
                     SYCLConv2DParams const& params, read_accessor const input,
                     write_accessor output)
      : out_offset_{out_offset},
        n_threads_{params.features_ * n_tile_cols * n_tile_rows *
                   params.batch_},
        n_tiles_{n_tiles},
        n_tile_rows_{n_tile_rows},
        n_tile_cols_{n_tile_cols},
        n_out_rows_{params.out_rows_},
        n_out_cols_{params.out_cols_},
        n_features_{params.features_},
        input_accessor_{input},
        output_accessor_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_threads_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      T* output_data =
          ConvertToActualTypeSycl(T, output_accessor_) + out_offset_;

      const helpers::TensorIndex2D tile_feature_idx =
          helpers::unflatten2d<Index, false>(index, n_features_, n_features_);
      const Index tile_idx = tile_feature_idx.s0;
      const Index feature = tile_feature_idx.s1;

      const helpers::TensorIndex3D tile_tensor_idx =
          helpers::unflatten3d<Index, false>(
              tile_idx, n_tile_rows_, n_tile_rows_, n_tile_cols_, n_tile_cols_);
      const Index col_idx = tile_tensor_idx.s2;
      const Index row_idx = tile_tensor_idx.s1;
      const Index batch = tile_tensor_idx.s0;

      IntermediateTile<T, M, N, R, S> tmp{input_data, tile_idx, n_tiles_,
                                          feature, n_features_};

      const Index col = col_idx * N;
      const Index cend = cl::sycl::min(col + N, n_out_cols_);

      const Index row = row_idx * M;
      const Index rend = cl::sycl::min(row + M, n_out_rows_);

      const Index offset =
          ((batch * n_out_rows_ + row) * n_out_cols_ + col) * n_features_ +
          feature;

      SYCLOutputWindow out_w{rend - row, cend - col, offset};

      OutputData<T, M, N, R, S>::write_output(output_data, out_w, n_out_cols_,
                                              n_features_,
                                              OutputTile<T, M, N, R, S>{tmp});
    }
  }

 private:
  const Index out_offset_;
  const Index n_threads_;
  const Index n_tiles_;
  const Index n_tile_rows_;
  const Index n_tile_cols_;
  const Index n_out_rows_;
  const Index n_out_cols_;
  const Index n_features_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T, int M, int N, int R, int S, bool accumulate_output>
struct ExtractOutputTiles<T, M, N, R, S, ConvType::FilterBackprop,
                          accumulate_output> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  ExtractOutputTiles(Index const /*n_tiles*/, SYCLConv2DParams const& params,
                     read_accessor const input, write_accessor output)
      : n_threads_{params.features_ * params.channels_},
        n_features_{params.features_},
        n_channels_{params.channels_},
        input_accessor_{input},
        output_accessor_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_threads_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const helpers::TensorIndex2D channel_feature_idx =
          helpers::unflatten2d<Index, false>(index, n_features_, n_features_);
      const Index channel = channel_feature_idx.s0;
      const Index feature = channel_feature_idx.s1;

      IntermediateTile<T, M, N, R, S> tmp{input_data, channel, n_channels_,
                                          feature, n_features_};
      OutputData<T, M, N, R, S>::template write_filter_output<
          accumulate_output>(output_data, channel, feature, n_channels_,
                             n_features_, OutputTile<T, M, N, R, S>{tmp});
    }
  }

 private:
  const Index n_threads_;
  const Index n_features_;
  const Index n_channels_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
}  // namespace winograd
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_KERNELS_H_
