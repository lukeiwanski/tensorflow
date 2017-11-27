#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_KERNELS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_KERNELS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_ops_sycl_common.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace im2col {

template <typename T, ConvType CType>
struct ExtractInputTiles;
/**
 * Have one thread per input entry. That thread is then responsible for writing
 * its one entry to each point in the intermediate tensor as required for the
 * contraction.
 */
template <typename T>
struct ExtractInputTiles<T, ConvType::Forward> {
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
      Index const n_items, Index const in_offset, Index const tile_size,
      SYCLConv2DParams const& params, read_accessor const input,
      write_accessor output)
      : n_items_{n_items},
        in_offset_{in_offset},
        tile_size_{tile_size},
        p_{params},
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get(0);
    if (index < n_items_) {
      const T* input_data =
          ConvertToActualTypeSycl(T, input_accessor_) + in_offset_;
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index channel = index % p_.channels_;
      const Index input_idx = index / p_.channels_;
      T in_val = input_data[index];
      const SYCL2DWindow w = p_.output_window_from_input(input_idx);

      for (Index r = w.rstart, in_r = p_.window_rows_ - 1 - w.firstr;
           r < w.rend; ++r, in_r -= p_.stride_rows_) {
        if (r >= 0) {
          for (Index c = w.cstart, in_c = p_.window_cols_ - 1 - w.firstc;
               c < w.cend; ++c, in_c -= p_.stride_cols_) {
            if (c >= 0) {
              T* tile_start =
                  output_data +
                  ((w.batch * p_.out_rows_ + r) * p_.out_cols_ + c) *
                      tile_size_;
              Index tile_idx =
                  (in_r * p_.window_cols_ + in_c) * p_.channels_ + channel;
              tile_start[tile_idx] = in_val;
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
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T>
struct ExtractInputTiles<T, ConvType::InputBackprop> {
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
      Index const n_items, Index const in_offset, Index const tile_size,
      SYCLConv2DParams const& params, read_accessor const input,
      write_accessor output)
      : n_items_{n_items},
        in_offset_{in_offset},
        tile_size_{tile_size},
        p_{params},
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get(0);
    if (index < n_items_) {
      T const* const input_data =
          ConvertToActualTypeSycl(T, input_accessor_) + in_offset_;
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index feature = index % p_.features_;
      const Index output_idx = index / p_.features_;
      const SYCL2DWindow w = p_.input_window_from_output(output_idx);
      T in_val = input_data[index];

      for (Index r = cl::sycl::max(w.rstart, 0),
                 in_r = p_.window_rows_ - 1 - w.firstr;
           r < w.rend; ++r, --in_r) {
        for (Index c = cl::sycl::max(w.cstart, 0),
                   in_c = p_.window_cols_ - 1 - w.firstc;
             c < w.cend; ++c, --in_c) {
          T* tile_start =
              output_data +
              ((w.batch * p_.in_rows_ + r) * p_.in_cols_ + c) * tile_size_;
          Index tile_idx =
              (in_r * p_.window_cols_ + in_c) * p_.features_ + feature;
          tile_start[tile_idx] = in_val;
        }
      }
    }
  }

 private:
  const Index n_items_;
  const Index in_offset_;
  const Index tile_size_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T>
struct ExtractInputTiles<T, ConvType::FilterBackprop> {
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
      Index const n_items, Index const in_offset, Index const tile_size,
      SYCLConv2DParams const& params, read_accessor const input,
      write_accessor output)
      : n_items_{n_items},
        in_offset_{in_offset},
        tile_size_{tile_size},
        p_{params},
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get(0);
    if (index < n_items_) {
      const T* input_data =
          ConvertToActualTypeSycl(T, input_accessor_) + in_offset_;
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index channel = index % p_.channels_;
      const Index input_idx = index / p_.channels_;
      T in_val = input_data[index];
      const SYCL2DWindow w = p_.output_window_from_input(input_idx);

      for (Index r = w.rstart, in_r = p_.window_rows_ - 1; r < w.rend;
           r += p_.dilation_rows_, in_r -= p_.stride_rows_) {
        if (r >= 0) {
          for (Index c = w.cstart, in_c = p_.window_cols_ - 1; c < w.cend;
               c += p_.dilation_cols_, in_c -= p_.stride_cols_) {
            if (c >= 0) {
              T* tile_start =
                  output_data +
                  ((r * p_.out_cols_ + c) * p_.channels_ + channel) *
                      tile_size_;
              Index tile_idx =
                  ((w.batch * p_.window_rows_ + in_r) * p_.window_cols_ + in_c);
              tile_start[tile_idx] = in_val;
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
  const SYCLConv2DParams p_;
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
      Index const n_items, Index const in_offset,
      SYCLConv2DParams const& params, read_accessor const input,
      write_accessor output)
      : n_items_{n_items},
        in_offset_{in_offset},
        p_{params},
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get(0);
    if (index < n_items_) {
      T const* const input_data =
          ConvertToActualTypeSycl(T, input_accessor_) + in_offset_;
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);
      T in_val = input_data[index];

      const Index feature = index % p_.features_;
      index /= p_.features_;
      const Index channel = index % p_.channels_;
      index /= p_.channels_;
      const Index col = index % p_.window_cols_;
      index /= p_.window_cols_;
      const Index row = index;

      const Index out_row = p_.window_rows_ - 1 - row;
      const Index out_col = p_.window_cols_ - 1 - col;
      const Index out_idx =
          ((out_row * p_.window_cols_ + out_col) * p_.features_ + feature) *
              p_.channels_ +
          channel;
      output_data[out_idx] = in_val;
    }
  }

 private:
  const Index n_items_;
  const Index in_offset_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
}  // namespace im2col
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_KERNELS_H_
