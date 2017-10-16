/*
 * Copyright 2017 John Lawson, Codeplay Software.
 * All rights reserved.
 */

#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_ops_sycl_common.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
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
      Index const n_threads, Index const in_offset, Index const tile_size,
      SYCLConv2DParams const& params, read_accessor const input,
      write_accessor output)
      : n_threads_{n_threads},
        in_offset_{in_offset},
        tile_size_{tile_size},
        p_{params},
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get(0);
    if (index < n_threads_) {
      const T* input_data =
          ConvertToActualTypeSycl(T, input_accessor_) + in_offset_;
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index channel = index % p_.channels_;
      const Index input_idx = index / p_.channels_;
      const SYCL2DWindow w = p_.output_window_from_input(input_idx);
      T in_val = input_data[index];

      for (Index r = w.rstart, in_r = p_.window_rows_ - 1 - w.firstr;
           r < w.rend; ++r, in_r -= p_.stride_rows_) {
        for (Index c = w.cstart, in_c = p_.window_cols_ - 1 - w.firstc;
             c < w.cend; ++c, in_c -= p_.stride_cols_) {
          T* tile_start =
              output_data +
              ((w.batch * p_.out_rows_ + r) * p_.out_cols_ + c) * tile_size_;
          Index tile_idx =
              (in_r * p_.window_cols_ + in_c) * p_.channels_ + channel;
          tile_start[tile_idx] = in_val;
        }
      }
    }
  }

 private:
  const Index n_threads_;
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
      Index const n_threads, Index const in_offset, Index const out_offset,
      SYCLConv2DParams const& params, read_accessor const input,
      write_accessor output)
      : n_threads_{n_threads},
        in_offset_{in_offset},
        out_offset_{out_offset},
        p_{params},
        input_accessor_{input},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get(0);
    if (index < n_threads_) {
      const T* input_data =
          ConvertToActualTypeSycl(T, input_accessor_) + in_offset_;
      T* output_data =
          ConvertToActualTypeSycl(T, output_accessor_) + out_offset_;

      const Index channel = index % p_.channels_;
      const Index tile_idx = index / p_.channels_;
      const SYCL2DWindow w = p_.output_window_from_input(tile_idx);

      const Index image_size = p_.in_cols_ * p_.in_rows_ * p_.channels_;
      T const* const input_data_start =
          input_data + w.batch * image_size + channel;
      const Index tile_size = p_.batch_ * p_.window_rows_ * p_.window_cols_;
      T* const output_data_start = output_data + tile_idx * tile_size + channel;
      const Index rsize = w.rend - w.rstart;
      const Index csize = w.cend - w.cstart;
      Index out_idx = 0;
      Index in_r_idx = w.rstart;
      for (Index r = 0; r < p_.window_rows_; ++r) {
        Index in_c_idx = w.cstart;
        for (Index c = 0; c < p_.window_cols_; ++c) {
          const Index idx = in_r_idx + in_c_idx;
          output_data_start[out_idx] =
              (r < w.firstr || c < w.firstc || r >= rsize || c >= csize)
                  ? static_cast<T>(0)
                  : input_data_start[idx];
          out_idx += p_.channels_;
          for (Index sc = 0; sc < p_.stride_cols_ && c < p_.window_cols_ - 1;
               ++sc) {
            // Fill column entries with stride zeros
            output_data_start[out_idx] = static_cast<T>(0);
            out_idx += p_.channels_;
          }
        }
        in_c_idx += p_.channels_;
        for (Index sr = 0; sr < p_.stride_rows_ && r < p_.window_rows_ - 1;
             ++sr) {
          // Fill a whole row with stride zeros
          for (Index sc = 0; sc < p_.window_cols_; ++sc) {
            output_data_start[out_idx] = static_cast<T>(0);
            out_idx += p_.channels_;
          }
        }
      }
      in_r_idx += p_.in_cols_;
    }
  }

 private:
  const Index n_threads_;
  const Index in_offset_;
  const Index out_offset_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
}
template <typename T, ConvType CType>
struct LaunchIm2Col;
template <typename T>
struct LaunchIm2Col<T, ConvType::Forward> {
  using Index = int;

  static void launch_input_transform(Eigen::SyclDevice const& device,
                                     T const* const input,
                                     Index const in_offset, T* const output,
                                     Index const n_tiles, Index const tile_size,
                                     const SYCLConv2DParams& params) {
    using Functor = im2col::ExtractInputTiles<T, ConvType::Forward>;
    static constexpr auto read_mode = Functor::read_mode;
    static constexpr auto write_mode = Functor::write_mode;

    const Index workgroup_size = device.maxSyclThreadsPerBlock();
    const Index n_items =
        params.batch_ * params.in_rows_ * params.in_cols_ * params.channels_;
    const Index n_threads = RoundUpToNearestMultiple(n_items, workgroup_size);

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access = device.get_sycl_accessor<read_mode>(cgh, input);
      auto transform_access = device.get_sycl_accessor<write_mode>(cgh, output);

      Functor extract_fun(n_items, in_offset, tile_size, params, input_access,
                          transform_access);
      cgh.parallel_for(cl::sycl::range<1>(n_threads), extract_fun);
    });
  }
  static void launch_matmul(Eigen::SyclDevice const& device,
                            T const* const input, T const* const filter,
                            T* const output, Index const n_tiles,
                            Index const tile_size, Index const n_features) {
    using ConstTensorType =
        Eigen::Tensor<T const, 2, Eigen::RowMajor, Eigen::DenseIndex>;
    using ConstTensor = Eigen::TensorMap<ConstTensorType, Eigen::Aligned>;
    using TensorType = Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>;
    using Tensor = Eigen::TensorMap<TensorType, Eigen::Aligned>;
    using TensorShape = Eigen::DSizes<Eigen::DenseIndex, 2>;
    using ContractDims = Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>;

    TensorShape const in_shape{n_tiles, tile_size};
    ConstTensor in_tensor{input, in_shape};
    TensorShape const fil_shape{tile_size, n_features};
    ConstTensor fil_tensor{filter, fil_shape};
    TensorShape const out_shape{n_tiles, n_features};
    Tensor out_tensor{output, out_shape};

    out_tensor.device(device) = in_tensor.contract(fil_tensor, ContractDims{});
  }
  static bool launch(OpKernelContext* context, Tensor* tensor_out,
                     const Tensor& tensor_in, const Tensor& tensor_filter,
                     const SYCLConv2DParams& params) {
    auto const& device = context->eigen_device<Eigen::SyclDevice>();
    cl::sycl::queue& sycl_queue = device.sycl_queue();
    cl::sycl::device const& sycl_device = sycl_queue.get_device();
    size_t const alloc_limit =
        sycl_device.get_info<cl::sycl::info::device::max_mem_alloc_size>();
    const Index n_tile_rows = params.out_rows_;
    const Index n_tile_cols = params.out_cols_;
    const Index tile_size =
        params.window_rows_ * params.window_cols_ * params.channels_;

    size_t alloc_size_per_image =
        n_tile_rows * n_tile_cols * tile_size * sizeof(T);
    // The number of images per alloc is bounded above by the total number of
    // images in a batch
    size_t images_per_alloc =
        std::min<size_t>(params.batch_, alloc_limit / alloc_size_per_image);

    size_t n_input_transforms = params.batch_ / images_per_alloc;
    size_t last_batch_size =
        params.batch_ - (n_input_transforms * images_per_alloc);
    if (last_batch_size > 0) {
      ++n_input_transforms;
    } else {
      last_batch_size = images_per_alloc;
    }
    T* const output = tensor_out->template flat<T>().data();
    std::vector<T*> temp_ptrs;
    temp_ptrs.reserve(n_input_transforms);
    for (int i = 0; i < n_input_transforms; ++i) {
      const Index in_offset = i * images_per_alloc * params.in_rows_ *
                              params.in_cols_ * params.channels_;
      const Index out_offset = i * images_per_alloc * params.out_rows_ *
                               params.out_cols_ * params.features_;
      if (i == n_input_transforms - 1) {
        images_per_alloc = last_batch_size;
      }
      const Index n_tiles = images_per_alloc * n_tile_rows * n_tile_cols;
      const Index transform_size = alloc_size_per_image * images_per_alloc;

      T* const transform = static_cast<T*>(device.allocate(transform_size));
      temp_ptrs.push_back(transform);
      device.memset(transform, 0, transform_size);

      T const* const input = tensor_in.template flat<T>().data();
      T const* const filter = tensor_filter.template flat<T>().data();
      launch_input_transform(device, input, in_offset, transform, n_tiles,
                             tile_size, params);
      launch_matmul(device, transform, filter, output + out_offset, n_tiles,
                    tile_size, params.features_);
    }
    for (auto&& tmp_ptr : temp_ptrs) {
      device.deallocate(tmp_ptr);
    }
    return true;
  }
};
/*
template <typename T, int M, int N, int R, int S>
struct LaunchIm2Col<T, M, N, R, S, ConvType::FilterBackprop> {
  using Index = int;
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  static constexpr auto CType = ConvType::FilterBackprop;

  static T* launch_input_transform(Eigen::SyclDevice const& device,
                                   const Tensor& tensor_in,
                                   Index const n_tile_rows,
                                   Index const n_tile_cols,
                                   const SYCLConv2DParams& params) {
    using Functor = im2col::ExtractInputTiles<T, M, N, R, S, CType>;
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
    using Functor = im2col::ExtractKernelTiles<T, M, N, R, S, CType>;
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

    im2col::BatchMatmul<T, 0, 0>()(device, in_tensor, fil_tensor, out_tensor);
    return output;
  }
  static void launch_output_transform(Eigen::SyclDevice const& device,
                                      T const* const input, Index const n_tiles,
                                      SYCLConv2DParams const& params,
                                      Tensor* const out) {
    using Functor = im2col::ExtractOutputTiles<T, M, N, R, S, CType>;
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
*/
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_H_
