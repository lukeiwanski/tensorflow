#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_SYCL_KERNEL_HELPERS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_SYCL_KERNEL_HELPERS_H_

#include "tensorflow/core/kernels/conv_ops_sycl_fast_div.h"
#include "tensorflow/core/kernels/conv_ops_sycl_kernel_macros.h"

namespace tensorflow {
namespace helpers {
/**
 * A 2D tensor index. The most packed index is s1, with s0 the least packed.
 */
struct TensorIndex2D {
  int s0;
  int s1;
};
/**
 * Compute a 2D tensor index from a flattened index. The most packed dimension
 * in memory is assumed to be the last one (i.e. the dimension with size_1
 * elements), while the size of the least packed dimension is not needed for the
 * calculation.
 */
template <typename Index, bool use_fast_div>
inline SNN_ALWAYS_INLINE TensorIndex2D
unflatten2d(Index index,
            typename fast_div::index_div<Index, use_fast_div>::type div_size_1,
            Index size_1) {
  const Index s01_idx = index;

  const Index s0 = s01_idx / div_size_1;
  const Index s1 = s01_idx - s0 * size_1;

  TensorIndex2D result{s0, s1};
  return result;
}
template <>
inline SNN_ALWAYS_INLINE TensorIndex2D unflatten2d<int, false>(
    int index, typename fast_div::index_div<int, false>::type /*div_size_1*/,
    int size_1) {
  const int s01_idx = index;

  const int s0 = s01_idx / size_1;
  const int s1 = s01_idx % size_1;

  TensorIndex2D result{s0, s1};
  return result;
}
/**
 * A 3D tensor index. The most packed index is s2, with s0 the least packed.
 */
struct TensorIndex3D {
  int s0;
  int s1;
  int s2;
  int s3;
};
/**
 * Compute a 3D tensor index from a flattened index. The most packed dimension
 * in memory is assumed to be the last one (i.e. the dimension with size_2
 * elements), while the size of the least packed dimension is not needed for the
 * calculation.
 */
template <typename Index, bool use_fast_div>
inline SNN_ALWAYS_INLINE TensorIndex3D
unflatten3d(Index index,
            typename fast_div::index_div<Index, use_fast_div>::type div_size_1,
            Index size_1,
            typename fast_div::index_div<Index, use_fast_div>::type div_size_2,
            Index size_2) {
  const Index s012_idx = index;

  const Index s01_idx = s012_idx / div_size_2;
  const Index s2 = s012_idx - s012_idx * size_2;
  const Index s0 = s01_idx / div_size_1;
  const Index s1 = s01_idx - s0 * size_1;

  TensorIndex3D result{s0, s1, s2};
  return result;
}
template <>
inline SNN_ALWAYS_INLINE TensorIndex3D unflatten3d<int, false>(
    int index, typename fast_div::index_div<int, false>::type /*div_size_1*/,
    int size_1, typename fast_div::index_div<int, false>::type /*div_size_2*/,
    int size_2) {
  const int s012_idx = index;

  const int s01_idx = s012_idx / size_2;
  const int s2 = s012_idx % size_2;
  const int s0 = s01_idx / size_1;
  const int s1 = s01_idx % size_1;

  TensorIndex3D result{s0, s1, s2};
  return result;
}
/**
 * A 4D tensor index. The most packed index is s3, with s0 the least packed.
 */
struct TensorIndex4D {
  int s0;
  int s1;
  int s2;
  int s3;
};
/**
 * Compute a 4D tensor index from a flattened index. The most packed dimension
 * in memory is assumed to be the last one (i.e. the dimension with size_3
 * elements), while the size of the least packed dimension is not needed for the
 * calculation.
 */
template <typename Index, bool use_fast_div>
inline SNN_ALWAYS_INLINE TensorIndex4D
unflatten4d(Index index,
            typename fast_div::index_div<Index, use_fast_div>::type div_size_1,
            Index size_1,
            typename fast_div::index_div<Index, use_fast_div>::type div_size_2,
            Index size_2,
            typename fast_div::index_div<Index, use_fast_div>::type div_size_3,
            Index size_3) {
  const Index s0123_idx = index;

  const Index s012_idx = s0123_idx / div_size_3;
  const Index s3 = s0123_idx - s012_idx * size_3;
  const Index s01_idx = s012_idx / div_size_2;
  const Index s2 = s012_idx - s01_idx * size_2;
  const Index s0 = s01_idx / div_size_1;
  const Index s1 = s01_idx - s0 * size_1;

  TensorIndex4D result{s0, s1, s2, s3};
  return result;
}
template <>
inline SNN_ALWAYS_INLINE TensorIndex4D unflatten4d<int, false>(
    int index, typename fast_div::index_div<int, false>::type /*div_size_1*/,
    int size_1, typename fast_div::index_div<int, false>::type /*div_size_2*/,
    int size_2, typename fast_div::index_div<int, false>::type /*div_size_3*/,
    int size_3) {
  const int s0123_idx = index;

  const int s012_idx = s0123_idx / size_3;
  const int s3 = s0123_idx % size_3;
  const int s01_idx = s012_idx / size_2;
  const int s2 = s012_idx % size_2;
  const int s0 = s01_idx / size_1;
  const int s1 = s01_idx % size_1;

  TensorIndex4D result{s0, s1, s2, s3};
  return result;
}
namespace io {
template <typename T>
struct Load {
  template <typename _T, typename Index>
  T SNN_ALWAYS_INLINE operator()(_T const* const ptr, Index const offset) {
    return ptr[offset];
  }
};
template <typename T, int N>
struct Load<cl::sycl::vec<T, N>> {
  template <typename _T, typename Index>
  cl::sycl::vec<T, N> SNN_ALWAYS_INLINE operator()(_T const* const ptr,
                                                   Index const offset) {
    static constexpr auto address_space =
        cl::sycl::access::address_space::global_space;
    // Surely there's a better way of loading a vector from a pointer. Surely.
    _T* non_const_ptr = const_cast<_T*>(ptr);
    cl::sycl::multi_ptr<T, address_space> mptr(non_const_ptr + offset);
    cl::sycl::vec<T, N> result;
    result.load(0, mptr);
    return result;
  }
};
template <typename T>
struct Load<cl::sycl::vec<T, 1>> {
  template <typename _T, typename Index>
  cl::sycl::vec<T, 1> SNN_ALWAYS_INLINE operator()(_T const* const ptr,
                                                   Index const offset) {
    cl::sycl::vec<T, 1> result(ptr[offset]);
    return result;
  }
};
template <typename T>
struct Store {
  template <typename _T, typename Index>
  void SNN_ALWAYS_INLINE operator()(_T* ptr, Index const offset, T const val) {
    ptr[offset] = val;
  }
};
template <typename T, int N>
struct Store<cl::sycl::vec<T, N>> {
  template <typename _T, typename Index>
  void SNN_ALWAYS_INLINE operator()(_T* ptr, Index const offset,
                                    cl::sycl::vec<T, N> const val) {
    static constexpr auto address_space =
        cl::sycl::access::address_space::global_space;
    cl::sycl::multi_ptr<T, address_space> mptr(ptr + offset);
    val.store(0, mptr);
  }
};
template <typename T>
struct Store<cl::sycl::vec<T, 1>> {
  template <typename _T, typename Index>
  void SNN_ALWAYS_INLINE operator()(_T* ptr, Index const offset,
                                    cl::sycl::vec<T, 1> val) {
    ptr[offset] = val.s0();
  }
};
}  // namespace io
namespace math {
template <typename T>
struct Mad {
  T operator()(T a, T b, T c) { return cl::sycl::mad(a, b, c); }
};
template <typename T>
struct Mad<cl::sycl::vec<T, 1>> {
  using VecType = cl::sycl::vec<T, 1>;
  VecType operator()(VecType a, VecType b, VecType c) {
    return VecType{cl::sycl::mad(a.s0(), b.s0(), c.s0())};
  }
};
template <typename T>
struct Dot {
  T operator()(T a, T b) { return a * b; }
};
template <typename T, int N>
struct Dot<cl::sycl::vec<T, N>> {
  using VecType = cl::sycl::vec<T, N>;
  static_assert(
      std::is_same<float, typename std::remove_cv<T>::type>::value ||
          std::is_same<double, typename std::remove_cv<T>::type>::value,
      "Dot product is only supported on floats and doubles.");
  static_assert(
      N == 2 || N == 3 || N == 4,
      "SYCL dot product is only valid for vector types with size 2, 3 or 4");
  T operator()(VecType a, VecType b) { return cl::sycl::dot(a, b); }
};
template <typename T>
struct Dot<cl::sycl::vec<T, 1>> {
  using VecType = cl::sycl::vec<T, 1>;
  T operator()(VecType a, VecType b) { return a.s0() * b.s0(); }
};
#ifndef __SYCL_DEVICE_ONLY__
// Work around a computecpp bug which doesn't provide an implementation of dot
// on the host for 2 element vectors.
template <typename T>
struct Dot<cl::sycl::vec<T, 2>> {
  using VecType = cl::sycl::vec<T, 2>;
  T operator()(VecType a, VecType b) {
    return a.s0() * b.s0() + a.s1() * b.s1();
  }
};
#endif
template <typename T>
struct Dot<cl::sycl::vec<T, 8>> {
  using HalfVecType = cl::sycl::vec<T, 4>;
  using VecType = cl::sycl::vec<T, 8>;
  T operator()(VecType a, VecType b) {
    return Dot<HalfVecType>()(HalfVecType{a.hi()}, HalfVecType{b.hi()}) +
           Dot<HalfVecType>()(HalfVecType{a.lo()}, HalfVecType{b.lo()});
  }
};
template <typename T>
struct Dot<cl::sycl::vec<T, 16>> {
  using HalfVecType = cl::sycl::vec<T, 8>;
  using VecType = cl::sycl::vec<T, 16>;
  T operator()(VecType a, VecType b) {
    return Dot<HalfVecType>()(HalfVecType{a.hi()}, HalfVecType{b.hi()}) +
           Dot<HalfVecType>()(HalfVecType{a.lo()}, HalfVecType{b.lo()});
  }
};
}  // namespace math
}  // namespace helpers
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_KERNEL_HELPERS_H_
