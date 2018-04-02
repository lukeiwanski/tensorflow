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

#ifndef TENSORFLOW_KERNELS_GATHER_FUNCTOR_H_
#define TENSORFLOW_KERNELS_GATHER_FUNCTOR_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace functor {

// Helper method to copy using memcpy.
template <typename T, typename Index, typename SliceIndex,
          SliceIndex static_slice_elems>
SliceIndex HandleCopies(OpKernelContext* ctx,
                        typename TTypes<T, 3>::ConstTensor params,
                        typename TTypes<Index>::ConstFlat indices,
                        SliceIndex slice_elems,
                        typename TTypes<T, 3>::Tensor out) {
  const SliceIndex indices_size = static_cast<SliceIndex>(indices.dimension(0));
  const SliceIndex batch_size = static_cast<SliceIndex>(params.dimension(0));
  const Index limit = static_cast<Index>(params.dimension(1));
  T* out_base = &out(0, 0, 0);
  const T* params_base = &params(0, 0, 0);
  if (static_slice_elems >= 0) {
    // Give compiler static knowledge of the number of elements/bytes
    slice_elems = static_slice_elems;
  }
  // Compute slice_bytes here so that static knowledge is available
  const size_t slice_bytes = slice_elems * sizeof(T);
  auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
  mutex mu;
  // Store the value of invalidate index for printing error information, it's a
  // shared variable.
  SliceIndex result = -1;
  auto work = [&](int64 start, int64 end) {
    SliceIndex batch_idx = static_cast<SliceIndex>(start / indices_size);
    SliceIndex indices_idx = static_cast<SliceIndex>(start % indices_size);
    SliceIndex batch_idx_end = static_cast<SliceIndex>(end / indices_size);
    SliceIndex indices_idx_end = static_cast<SliceIndex>(end % indices_size);

    while ((batch_idx < batch_idx_end) ||
           (batch_idx == batch_idx_end && indices_idx < indices_idx_end)) {
      SliceIndex i_next = indices_idx + 1;
      SliceIndex b_next = batch_idx + 1;
      if ((batch_idx == batch_idx_end && i_next < indices_idx_end) ||
          (i_next < indices_size)) {
        port::prefetch<port::PREFETCH_HINT_T0>(
            &params(batch_idx, indices(i_next), 0));
        port::prefetch<port::PREFETCH_HINT_T0>(&out(batch_idx, i_next, 0));
        b_next = batch_idx;
      } else if (b_next <= batch_idx_end) {
        port::prefetch<port::PREFETCH_HINT_T0>(&params(b_next, indices(0), 0));
        port::prefetch<port::PREFETCH_HINT_T0>(&out(b_next, 0, 0));
        i_next = 0;
      }
      const Index index = internal::SubtleMustCopy(indices(indices_idx));
      if (!FastBoundsCheck(index, limit)) {
        mutex_lock l(mu);
        result = indices_idx;
        return;
      }
      // Copy using memcpy if possible, otherwise an Eigen loop
      // TODO(cwhipkey): avoid linking to framework to get Allocator (to improve
      // ahead-of-time compilation binary size).
      if (is_simple_type<T>::value) {
        // Avoid auto-promotion to Index from SliceIndex by casting.
        memcpy(
            out_base + (batch_idx * indices_size + indices_idx) * slice_elems,
            params_base + (batch_idx * static_cast<SliceIndex>(limit) +
                           static_cast<SliceIndex>(index)) *
                              slice_elems,
            slice_bytes);
      } else {
        // For non-"simple" types (e.g. strings).
        out.template chip<1>(indices_idx) = params.template chip<1>(index);
      }
      indices_idx = i_next;
      batch_idx = b_next;
    }
  };

  Shard(worker_threads->num_threads, worker_threads->workers,
        batch_size * indices_size, slice_elems * sizeof(T), work);
  return result;
}

template <typename T, typename Index>
struct GatherFunctorCPU {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 3>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 3>::Tensor out) {
    const int64 N = indices.size();
    const int64 slice_size = out.dimension(2);
    int64 bad_i;

    bool use_large = (slice_size > std::numeric_limits<int32>::max() ||
                      params.size() > std::numeric_limits<int32>::max() ||
                      N > std::numeric_limits<int32>::max());
#define CALL(elems)                                                      \
  do {                                                                   \
    if (use_large) {                                                     \
      bad_i = HandleCopies<T, Index, int64, elems>(ctx, params, indices, \
                                                   slice_size, out);     \
    } else {                                                             \
      const int32 small_slice = static_cast<int32>(slice_size);          \
      bad_i = HandleCopies<T, Index, int32, elems>(ctx, params, indices, \
                                                   small_slice, out);    \
    }                                                                    \
  } while (0)

    if (slice_size == 10)
      CALL(10);
    else if (slice_size == 20)
      CALL(20);
    else
      CALL(-1);
#undef CALL

    return bad_i;
  }
};

template <typename Device, typename T, typename Index>
struct GatherFunctor {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 3>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 3>::Tensor out);
};

template <typename T, typename Index>
struct GatherFunctor<CPUDevice, T, Index> {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 3>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 3>::Tensor out) {
    return GatherFunctorCPU<T, Index>()(ctx, params, indices, out);
  }
};

#ifdef TENSORFLOW_USE_SYCL
// GatherOp is a very generic operation that extracts and copies a submatrix.
// It can be broken down in several more optimal sub-cases which is what this version is doing.
// When copying rows:
//   - copying a continuous chunk of memory (one or more rows) uses memcpy
// When copying columns:
//   - copying one column uses chip
//   - copying several adjacent columns uses slice
//   - copying the same column several times uses chip and broadcast
//   - generic case uses several chips
template <typename T, typename Index>
struct HandleCopiesDetails {
  using ParamsT = typename TTypes<T, 3>::ConstTensor;
  using IndicesT = typename TTypes<Index>::ConstFlat;
  using OutT = typename TTypes<T, 3>::Tensor;
  using TensorDataT = T;
  using IndicesDataT = Index;
  using TensorIndexT = typename ParamsT::Index;

  HandleCopiesDetails(const SYCLDevice& d,
                      ParamsT p,
                      IndicesT i,
                      OutT o)
    : device(d),
      params(p),
      indices(i),
      out(o),
      indices_size(i.dimension(0)),
      limit(p.dimension(1)),
      slice_size(0),
      host_indices() {}

  virtual ~HandleCopiesDetails() = default;

  virtual void full_copy(TensorIndexT from) = 0;
  virtual void adjacent_copy(TensorIndexT from, TensorIndexT to) = 0;
  virtual void broadcast_copy(TensorIndexT from, TensorIndexT to) = 0;
  virtual void single_copy(TensorIndexT from) = 0;

  const SYCLDevice& device;
  ParamsT params;
  IndicesT indices;
  OutT out;
  TensorIndexT indices_size;
  TensorIndexT limit;
  TensorIndexT slice_size;
  std::vector<IndicesDataT> host_indices;
};

template <typename T, typename Index>
struct HandleCopiesDetailsRows : public HandleCopiesDetails<T, Index> {
  using Parent = HandleCopiesDetails<T, Index>;
  using TensorIndexT = typename Parent::TensorIndexT;

  HandleCopiesDetailsRows(const SYCLDevice& d,
                          typename Parent::ParamsT p,
                          typename Parent::IndicesT i,
                          typename Parent::OutT o)
    : Parent(d, p, i, o) {
    this->slice_size = o.dimension(2);
  }

  virtual inline void full_copy(TensorIndexT from) override {
    this->device.memcpy(this->out.data(),
                        this->params.data() + (from * this->slice_size),
                        this->slice_size * sizeof(T));
  }

  virtual inline void adjacent_copy(TensorIndexT from, TensorIndexT to) override {
    this->device.memcpy(this->out.data() + (from * this->slice_size),
                        this->params.data() + (this->host_indices[from] * this->slice_size),
                        (to - from) * this->slice_size * sizeof(T));
  }

  virtual inline void broadcast_copy(TensorIndexT from, TensorIndexT to) override {
    // On some device, it may be more efficient to launch a chip and a
    // broadcast instead of several memcpy
    for (TensorIndexT i = from; i < to; ++i) {
      this->single_copy(i);
    }
  }

  virtual inline void single_copy(TensorIndexT from) override {
    this->adjacent_copy(from, from + 1);
  }
};

template <typename T, typename Index>
struct HandleCopiesDetailsCols : public HandleCopiesDetails<T, Index> {
  using Parent = HandleCopiesDetails<T, Index>;
  using TensorIndexT = typename Parent::TensorIndexT;

  HandleCopiesDetailsCols(const SYCLDevice& d,
                          typename Parent::ParamsT p,
                          typename Parent::IndicesT i,
                          typename Parent::OutT o)
    : Parent(d, p, i, o),
      out_slice_offsets({0, 0, 0}),
      params_slice_offsets({0, 0, 0}),
      out_slice_extents({1, 1, 1}),
      params_slice_extents({1, 1, 1}),
      bcast_shape({1, 1, 1}) {
      this->slice_size = o.dimension(0);
      out_slice_extents[0] = this->slice_size;
      params_slice_extents[0] = this->slice_size;
      out_slice_extents[2] = o.dimension(2);
      params_slice_extents[2] = o.dimension(2);
    }

  virtual inline void full_copy(TensorIndexT from) override {
    this->out.template chip<1>(0).device(this->device) = this->params.template chip<1>(from);
  }

  virtual inline void adjacent_copy(TensorIndexT from, TensorIndexT to) override {
    out_slice_offsets[1] = from;
    params_slice_offsets[1] = this->host_indices[from];
    out_slice_extents[1] = to - from;
    params_slice_extents[1] = to - from;
    this->out.slice(out_slice_offsets, out_slice_extents).device(this->device) =
      this->params.slice(params_slice_offsets, params_slice_extents);
  }

  virtual inline void broadcast_copy(TensorIndexT from, TensorIndexT to) override {
    out_slice_offsets[1] = from;
    out_slice_extents[1] = to - from;
    bcast_shape[1] = to - from;
    this->out.template slice(out_slice_offsets, out_slice_extents).device(this->device) =
      this->params.template chip<1>(this->host_indices[from]).broadcast(bcast_shape);
  }

  virtual inline void single_copy(TensorIndexT from) override {
    this->out.template chip<1>(from).device(this->device) =
      this->params.template chip<1>(this->host_indices[from]);
  }

 private:
  Eigen::DSizes<TensorIndexT, 3> out_slice_offsets;
  Eigen::DSizes<TensorIndexT, 3> params_slice_offsets;
  Eigen::DSizes<TensorIndexT, 3> out_slice_extents;
  Eigen::DSizes<TensorIndexT, 3> params_slice_extents;
  Eigen::array<TensorIndexT, 3> bcast_shape;
};

template <typename HandleCopiesImplDetails>
typename HandleCopiesImplDetails::TensorIndexT
HandleCopiesSYCL(OpKernelContext* ctx,
                 typename HandleCopiesImplDetails::ParamsT params,
                 typename HandleCopiesImplDetails::IndicesT indices,
                 typename HandleCopiesImplDetails::OutT out) {
  HandleCopiesImplDetails impl_details(ctx->eigen_sycl_device(), params, indices, out);

  using TensorIndexT = typename HandleCopiesImplDetails::TensorIndexT;
  using IndicesDataT = typename HandleCopiesImplDetails::IndicesDataT;

  // Handle simple case
  if (impl_details.indices_size == 1) {
    const IndicesDataT index = internal::SubtleMustCopy(indices(0));
    if (!FastBoundsCheck(index, impl_details.limit))
      return 0;
    impl_details.full_copy(index);
    return -1;
  }

  // Grab the index and check its validity.  An earlier version of the
  // code checked it and then grabbed it from memory a second time, which
  // was a security risk since it could have changed in between.
  // Now copy the indices on the host.
  auto& host_indices = impl_details.host_indices;
  host_indices.reserve(impl_details.indices_size);
  for (TensorIndexT i = 0; i < impl_details.indices_size; i++) {
    const IndicesDataT index = internal::SubtleMustCopy(indices(i));
    if (!FastBoundsCheck(index, impl_details.limit)) {
      return i;
    }
    host_indices.push_back(index);
  }

  TensorIndexT copy_from = 0;  // bound included
  TensorIndexT copy_to = 1;    // bound excluded
  do {
    // Find block of incrementing indices to copy all the lines in one go
    while (copy_to < impl_details.indices_size &&
           host_indices[copy_to - 1] + 1 == host_indices[copy_to]) {
      ++copy_to;
    }
    if (copy_to > copy_from + 1) {
      // At least 2 consecutives indices were found
      impl_details.adjacent_copy(copy_from, copy_to);
    }
    else {
      // Find block of same indicies to broadcast lines
      while (copy_to < impl_details.indices_size &&
             host_indices[copy_to - 1] == host_indices[copy_to]) {
        ++copy_to;
      }
      if (copy_to > copy_from + 1) {
        // At least 2 same consecutive indices were found
        impl_details.broadcast_copy(copy_from, copy_to);
      }
      else {  // Generic copy of a single line
        impl_details.single_copy(copy_from);
      }
    }
    copy_from = copy_to++;
  } while (copy_to <= impl_details.indices_size);
  return -1;
}

template <typename T, typename Index>
struct GatherFunctorSYCL {
  inline int64 operator()(OpKernelContext* ctx,
                          typename TTypes<T, 3>::ConstTensor params,
                          typename TTypes<Index>::ConstFlat indices,
                          typename TTypes<T, 3>::Tensor out) {
    if (params.dimension(0) == 1)
      return HandleCopiesSYCL<HandleCopiesDetailsRows<T, Index>>(ctx, params, indices, out);
    else
      return HandleCopiesSYCL<HandleCopiesDetailsCols<T, Index>>(ctx, params, indices, out);
  }
};

// Stripped down version of CPU functor which uses Eigen loops to copy slices
template <typename T, typename Index>
struct GatherFunctor<SYCLDevice, T, Index> {
  inline int64 operator()(OpKernelContext* ctx,
                          typename TTypes<T, 3>::ConstTensor params,
                          typename TTypes<Index>::ConstFlat indices,
                          typename TTypes<T, 3>::Tensor out) {
    return GatherFunctorSYCL<T, Index>()(ctx, params, indices, out);
  }
};
#endif  // TENSORFLOW_USE_SYCL

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_GATHER_FUNCTOR_H_
