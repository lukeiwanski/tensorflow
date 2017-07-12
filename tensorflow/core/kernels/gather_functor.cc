/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/gather_functor.h"

#if GOOGLE_CUDA || defined(TENSORFLOW_USE_SYCL)
#include "tensorflow/core/framework/register_types.h"
#endif  // GOOGLE_CUDA || defined(TENSORFLOW_USE_SYCL)

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace functor {

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
#define DECLARE_GPU_SPECS_INDEX(T, Index)                             \
  template <>                                                         \
  int64 GatherFunctor<GPUDevice, T, Index>::operator()(               \
      const GPUDevice& d, typename TTypes<T, 3>::ConstTensor Tparams, \
      typename TTypes<Index>::ConstFlat Tindices,                     \
      typename TTypes<T, 3>::Tensor Tout);                            \
  extern template struct GatherFunctor<GPUDevice, T, Index>;

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64)

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
TF_CALL_complex64(DECLARE_GPU_SPECS);
TF_CALL_complex128(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
// Stripped down version of CPU functor which uses Eigen loops to copy slices
template <typename T, typename Index>
struct GatherFunctor<SYCLDevice, T, Index> {
  int64 operator()(const SYCLDevice& d, typename TTypes<T>::ConstMatrix params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T>::Matrix out) {
    const Index first_dim_size = static_cast<Index>(indices.dimension(0));
    const Index limit = static_cast<Index>(params.dimension(0));

    for (Index i = 0; i < first_dim_size; i++) {
      // Grab the index and check its validity.  An earlier version of the
      // code checked it and then grabbed it from memory a second time, which
      // was a security risk since it could have changed in between.
      const Index index = internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      out.template chip<0>(i).device(d) = params.template chip<0>(index);
    }
    return -1;
  }
};

// Force specialisations
#define DECLARE_GPU_SPECS_INDEX(T, Index) \
  template struct GatherFunctor<SYCLDevice, T, Index>;

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64)

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
#endif  // TENSORFLOW_USE_SYCL

}  // namespace functor
}  // namespace tensorflow
