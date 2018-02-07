/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Specialization of SpaceToBatchFunctor for a CPUDevice.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/spacetobatch_functor.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace functor {

namespace {

// Implementation of nested loops for SpaceToBatchOpFunctor.
//
// To simplify template implementation given lack of constexpr if, both the
// input and output pointers are non-const.
template <int N, bool B2S>
struct SpaceToBatchHelper {
  template <typename T>
  static void run(T* space_tensor_ptr, const int64* space_tensor_shape,
                  const int64* space_tensor_strides, const int64* block_shape,
                  const int64* pad_start, const int64* block_offsets,
                  const int64* batch_tensor_shape,
                  const int64* batch_tensor_strides, T* batch_tensor_ptr) {
    for (int64 batch_tensor_pos = 0; batch_tensor_pos < batch_tensor_shape[0];
         ++batch_tensor_pos) {
      const int64 space_tensor_pos =
          batch_tensor_pos * block_shape[0] + block_offsets[0] - pad_start[0];
      if (space_tensor_pos >= 0 && space_tensor_pos < space_tensor_shape[0]) {
        SpaceToBatchHelper<N - 1, B2S>::run(
            space_tensor_ptr + space_tensor_pos * space_tensor_strides[0],
            space_tensor_shape + 1, space_tensor_strides + 1, block_shape + 1,
            pad_start + 1, block_offsets + 1, batch_tensor_shape + 1,
            batch_tensor_strides + 1, batch_tensor_ptr);
      } else {
        if (B2S == false) {
          // Copy in padding.
          for (int64 i = 0; i < batch_tensor_strides[0]; ++i) {
            batch_tensor_ptr[i] = static_cast<T>(0);
          }
        }
      }
      batch_tensor_ptr += batch_tensor_strides[0];
    }
  }
};

template <bool B2S>
struct SpaceToBatchHelper<0, B2S> {
  template <typename T>
  static void run(T* space_tensor_ptr, const int64* space_tensor_shape,
                  const int64* space_tensor_strides, const int64* block_shape,
                  const int64* pad_start, const int64* block_offsets,
                  const int64* batch_tensor_shape,
                  const int64* batch_tensor_strides, T* batch_tensor_ptr) {
    for (int64 i = 0; i < batch_tensor_strides[-1]; ++i) {
      if (B2S == false) {
        batch_tensor_ptr[i] = space_tensor_ptr[i];
      } else {
        space_tensor_ptr[i] = batch_tensor_ptr[i];
      }
    }
  }
};

}  // namespace

template <typename T, int NUM_BLOCK_DIMS, bool B2S>
struct SpaceToBatchFunctor<CPUDevice, T, NUM_BLOCK_DIMS, B2S> {
  using SpaceT = typename std::conditional<B2S, T, const T>::type;
  using BatchT = typename std::conditional<B2S, const T, T>::type;
  Status operator()(
      const CPUDevice& d,
      typename TTypes<SpaceT, NUM_BLOCK_DIMS + 2>::Tensor space_tensor,
      const int64 block_shape_tensor[NUM_BLOCK_DIMS],
      const int64 paddings_tensor[NUM_BLOCK_DIMS * 2],
      typename TTypes<BatchT, NUM_BLOCK_DIMS + 2>::Tensor batch_tensor) {
    const int64 batch_tensor_batch = batch_tensor.dimension(0);

    const int64 space_tensor_batch = space_tensor.dimension(0);

    // Copy into local array so that the compiler is free to place in a
    // register.
    int64 pad_start[NUM_BLOCK_DIMS];
    int64 block_shape[NUM_BLOCK_DIMS];
    int64 space_tensor_shape[NUM_BLOCK_DIMS],
        batch_tensor_shape[NUM_BLOCK_DIMS];
    for (int block_dim = 0; block_dim < NUM_BLOCK_DIMS; ++block_dim) {
      pad_start[block_dim] = paddings_tensor[block_dim * 2];
      block_shape[block_dim] = block_shape_tensor[block_dim];
      space_tensor_shape[block_dim] = space_tensor.dimension(block_dim + 1);
      batch_tensor_shape[block_dim] = batch_tensor.dimension(block_dim + 1);
    }

    int64 space_tensor_strides[NUM_BLOCK_DIMS + 2],
        batch_tensor_strides[NUM_BLOCK_DIMS + 2];
    space_tensor_strides[NUM_BLOCK_DIMS + 1] =
        batch_tensor_strides[NUM_BLOCK_DIMS + 1] = 1;
    for (int dim = NUM_BLOCK_DIMS; dim >= 0; --dim) {
      space_tensor_strides[dim] =
          space_tensor_strides[dim + 1] * space_tensor.dimension(dim + 1);
      batch_tensor_strides[dim] =
          batch_tensor_strides[dim + 1] * batch_tensor.dimension(dim + 1);
    }

    // Use non-const pointers for both input and output to simplify template
    // implementation given lack of constexpr if.
    T* space_tensor_ptr = const_cast<T*>(space_tensor.data());
    T* batch_tensor_ptr = const_cast<T*>(batch_tensor.data());

    for (int64 batch_tensor_b = 0; batch_tensor_b < batch_tensor_batch;
         ++batch_tensor_b) {
      const int64 space_tensor_b = batch_tensor_b % space_tensor_batch;
      int64 block_index = batch_tensor_b / space_tensor_batch;
      int64 block_offsets[NUM_BLOCK_DIMS];
      for (int block_dim = NUM_BLOCK_DIMS - 1; block_dim >= 0; --block_dim) {
        // Skip unnecessary remainder operation for block_dim == 0.
        block_offsets[block_dim] =
            block_dim > 0 ? block_index % block_shape[block_dim] : block_index;
        block_index /= block_shape[block_dim];
      }

      // The compiler should inline the nested loops generated by this template.
      SpaceToBatchHelper<NUM_BLOCK_DIMS, B2S>::run(
          space_tensor_ptr + space_tensor_b * space_tensor_strides[0],
          space_tensor_shape, &space_tensor_strides[1], block_shape, pad_start,
          block_offsets, batch_tensor_shape, &batch_tensor_strides[1],
          batch_tensor_ptr + batch_tensor_b * batch_tensor_strides[0]);
    }
    return Status::OK();
  }
};

#ifdef TENSORFLOW_USE_SYCL

template <int NUM_BLOCK_DIMS>
struct S2BParameters {
  int32 space_tensor_batch;
  int32 batch_tensor_shape[NUM_BLOCK_DIMS + 2];
  int32 space_tensor_spatial_shape[NUM_BLOCK_DIMS];
  int32 pad_start[NUM_BLOCK_DIMS];
  int32 block_shape[NUM_BLOCK_DIMS];
};

template <typename T, int NUM_BLOCK_DIMS, bool B2S>
struct S2BKernel {
  using read_write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>;

  S2BKernel(read_write_accessor space_tensor_ptr,
            S2BParameters<NUM_BLOCK_DIMS> args,
            read_write_accessor batch_tensor_ptr, const int64 num_indices)
      : space_tensor_ptr_(space_tensor_ptr),
        batch_tensor_ptr_(batch_tensor_ptr),
        args_(args),
        num_indices_(num_indices) {}

  void operator()(cl::sycl::item<1> id) {
    T* space_tensor_ptr = ConvertToActualTypeSycl(T, space_tensor_ptr_);
    T* batch_tensor_ptr = ConvertToActualTypeSycl(T, batch_tensor_ptr_);

    for (int batch_tensor_idx = 0; batch_tensor_idx < num_indices_;
         batch_tensor_idx++) {
      int32 remaining_batch_tensor_idx = batch_tensor_idx;

      int32 batch_tensor_pos[NUM_BLOCK_DIMS + 2];

      for (int dim = NUM_BLOCK_DIMS + 1; dim >= 1; --dim) {
        batch_tensor_pos[dim] =
            remaining_batch_tensor_idx % args_.batch_tensor_shape[dim];
        remaining_batch_tensor_idx /= args_.batch_tensor_shape[dim];
      }
      batch_tensor_pos[0] = remaining_batch_tensor_idx;

      int32 remaining_block_idx =
          batch_tensor_pos[0] / args_.space_tensor_batch;
      int32 space_tensor_idx = batch_tensor_pos[NUM_BLOCK_DIMS + 1];
      int32 space_tensor_stride = args_.batch_tensor_shape[NUM_BLOCK_DIMS + 1];
      const int32 space_tensor_batch_pos =
          batch_tensor_pos[0] % args_.space_tensor_batch;
      for (int block_dim = NUM_BLOCK_DIMS - 1; block_dim >= 0; --block_dim) {
        int32 offset = remaining_block_idx;
        if (block_dim > 0) {
          offset %= args_.block_shape[block_dim];
        }
        int32 space_tensor_pos =
            batch_tensor_pos[block_dim + 1] * args_.block_shape[block_dim] +
            offset - args_.pad_start[block_dim];
        if (space_tensor_pos < 0 ||
            space_tensor_pos >= args_.space_tensor_spatial_shape[block_dim]) {
          if (B2S == false) {
            // In the space-to-batch case, write zero padding.
            batch_tensor_ptr[batch_tensor_idx] = static_cast<T>(0);
          }
          break;
        }
        space_tensor_idx += space_tensor_stride * space_tensor_pos;
        space_tensor_stride *= args_.space_tensor_spatial_shape[block_dim];
        if (block_dim == 0) {
          space_tensor_idx += space_tensor_stride * space_tensor_batch_pos;
          if (B2S == false) {
            batch_tensor_ptr[batch_tensor_idx] =
                space_tensor_ptr[space_tensor_idx];
          } else {
            space_tensor_ptr[space_tensor_idx] =
                batch_tensor_ptr[batch_tensor_idx];
          }
        }
        remaining_block_idx /= args_.block_shape[block_dim];
      }
    }
  }
  read_write_accessor space_tensor_ptr_;
  read_write_accessor batch_tensor_ptr_;
  S2BParameters<NUM_BLOCK_DIMS> args_;
  const int64 num_indices_;
};

template <typename T, int NUM_BLOCK_DIMS, bool B2S>
struct SpaceToBatchFunctor<SYCLDevice, T, NUM_BLOCK_DIMS, B2S> {
  using SpaceT = typename std::conditional<B2S, T, const T>::type;
  using BatchT = typename std::conditional<B2S, const T, T>::type;
  Status operator()(
      const SYCLDevice& d,
      typename TTypes<SpaceT, NUM_BLOCK_DIMS + 2>::Tensor space_tensor,
      const int64 block_shape[NUM_BLOCK_DIMS],
      const int64 paddings[NUM_BLOCK_DIMS * 2],
      typename TTypes<BatchT, NUM_BLOCK_DIMS + 2>::Tensor batch_tensor) {
    // Kernel execution fails if number of elements is zero.
    if (batch_tensor.size() == 0) {
      return Status::OK();
    }
    S2BParameters<NUM_BLOCK_DIMS> args;
    args.space_tensor_batch = space_tensor.dimension(0);
    for (int block_dim = 0; block_dim < NUM_BLOCK_DIMS; ++block_dim) {
      if (block_shape[block_dim] > std::numeric_limits<int32>::max()) {
        return errors::InvalidArgument("block_shape value exceeds 2^32-1");
      }
      args.block_shape[block_dim] = block_shape[block_dim];
      if (space_tensor.dimension(block_dim + 1) >
          std::numeric_limits<int32>::max()) {
        return errors::InvalidArgument("space_tensor dimension exceeds 2^32-1");
      }
      args.space_tensor_spatial_shape[block_dim] =
          space_tensor.dimension(block_dim + 1);
      if (paddings[block_dim * 2] > std::numeric_limits<int32>::max()) {
        return errors::InvalidArgument("paddings/crops value exceeds 2^32-1");
      }
      args.pad_start[block_dim] = paddings[block_dim * 2];
    }
    int64 total_count = 1;
    for (int dim = 0; dim < NUM_BLOCK_DIMS + 2; ++dim) {
      args.batch_tensor_shape[dim] = batch_tensor.dimension(dim);
      total_count *= args.batch_tensor_shape[dim];
    }
    if (total_count > std::numeric_limits<int32>::max()) {
      return errors::InvalidArgument(
          "number of batch_tensor elements exceeds 2^32-1");
    }

    const int num_threads = static_cast<int32>(total_count);
    auto space_tensor_buffer = d.get_sycl_buffer(space_tensor.data());
    auto batch_tensor_buffer = d.get_sycl_buffer(batch_tensor.data());

    d.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto space_tensor_acc =
          space_tensor_buffer
              .template get_access<cl::sycl::access::mode::read_write>(cgh);
      auto batch_tensor_acc =
          batch_tensor_buffer
              .template get_access<cl::sycl::access::mode::read_write>(cgh);

      S2BKernel<T, NUM_BLOCK_DIMS, B2S> kernel(space_tensor_acc, args,
                                               batch_tensor_acc, num_threads);

      cgh.parallel_for(cl::sycl::range<1>(num_threads), kernel);
    });

    return Status::OK();
  }
};
#endif  // TENSORFLOW_USE_SYCL

// Instantiate.
#ifdef TENSORFLOW_USE_SYCL
#define INSTANTIATE(NUM_BLOCK_DIMS, T)                                       \
  template struct SpaceToBatchFunctor<SYCLDevice, T, NUM_BLOCK_DIMS, false>; \
  template struct SpaceToBatchFunctor<SYCLDevice, T, NUM_BLOCK_DIMS, true>;  \
  template struct SpaceToBatchFunctor<CPUDevice, T, NUM_BLOCK_DIMS, false>;  \
  template struct SpaceToBatchFunctor<CPUDevice, T, NUM_BLOCK_DIMS, true>;
#else
#define INSTANTIATE(NUM_BLOCK_DIMS, T)                                      \
  template struct SpaceToBatchFunctor<CPUDevice, T, NUM_BLOCK_DIMS, false>; \
  template struct SpaceToBatchFunctor<CPUDevice, T, NUM_BLOCK_DIMS, true>;
#endif
  /**/

#define INSTANTIATE_FOR_T(T) \
  TF_SPACETOBATCH_FOR_EACH_NUM_BLOCK_DIMS(INSTANTIATE, T)

TF_CALL_REAL_NUMBER_TYPES(INSTANTIATE_FOR_T)

#undef INSTANTIATE_FOR_T
#undef INSTANTIATE

}  // namespace functor
}  // end namespace tensorflow
