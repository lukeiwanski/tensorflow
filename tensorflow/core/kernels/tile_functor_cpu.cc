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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/tile_functor.h"

namespace tensorflow {

namespace functor {
// Need to use a function object so we can provide partial specializations for
// different devices.
template <typename Device, typename T>
struct TileFunctor {
  void operator()(const Device& d, Tensor* out, const Tensor& in);
};
}

namespace internal {

template <typename Device, typename T>
void TileSimple(const Device& d, Tensor* out, const Tensor& in) {
  functor::TileFunctor<Device, T> functor;
  functor(d, out, in);
}

}  // end namespace internal

namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
struct TileFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, Tensor* out, const Tensor& in) {
    const int ndims = in.dims();
    const int64 nelem = out->NumElements();
    gtl::InlinedVector<int64, 8> in_strides = ComputeStride<int64>(in.shape());
    gtl::InlinedVector<int64, 8> out_strides =
        ComputeStride<int64>(out->shape());
    const T* p = in.flat<T>().data();
    T* q = out->flat<T>().data();

    for (int64 o_idx = 0; o_idx < nelem; ++o_idx) {
      int64 i_idx = 0;
      int64 t = o_idx;
      for (int i = 0; i < ndims; ++i) {
        i_idx += t / out_strides[i] % in.dim_size(i) * in_strides[i];
        t %= out_strides[i];
      }
      q[o_idx] = p[i_idx];
    }
  }
};

// Register functors used for Tile functor.
#define DEFINE_TYPE(T)                       \
  template struct Tile<CPUDevice, T, int32>; \
  template struct Tile<CPUDevice, T, int64>;

TF_CALL_bool(DEFINE_TYPE);
TF_CALL_float(DEFINE_TYPE);
TF_CALL_double(DEFINE_TYPE);
TF_CALL_uint8(DEFINE_TYPE);
TF_CALL_int32(DEFINE_TYPE);
TF_CALL_int16(DEFINE_TYPE);
TF_CALL_int64(DEFINE_TYPE);
TF_CALL_half(DEFINE_TYPE);
TF_CALL_complex64(DEFINE_TYPE);
TF_CALL_complex128(DEFINE_TYPE);
TF_CALL_string(DEFINE_TYPE);

#undef DEFINE_TYPE

#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
// SYCL kernel to copy tiles from the input to the output. Expects the number of
// threads to be the number of elements in the output tensor.
template <typename T>
struct TileSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;
  using stride_read_accessor =
      cl::sycl::accessor<int64, 2, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;
  TileSYCL(int ndims, const read_accessor input,
           const stride_read_accessor strides, write_accessor output)
      : ndims_(ndims),
        input_accessor_(input),
        stride_accessor_(strides),
        output_accessor_(output) {}

  void operator()(cl::sycl::item<1> id) {
    const T* input = ConvertToActualTypeSycl(T, input_accessor_);
    T* output = ConvertToActualTypeSycl(T, output_accessor_);

    const int64 o_idx = id.get_linear_id();
    int64 i_idx = 0;
    int64 t = o_idx;
    for (int i = 0; i < ndims_; ++i) {
      i_idx += ((t / stride_accessor_[cl::sycl::id<2>(1, i)]) %
                stride_accessor_[cl::sycl::id<2>(2, i)]) *
               stride_accessor_[cl::sycl::id<2>(0, i)];
      t %= stride_accessor_[cl::sycl::id<2>(1, i)];
    }
    output[o_idx] = input[i_idx];
  }

 private:
  const int ndims_;
  const read_accessor input_accessor_;
  const stride_read_accessor stride_accessor_;
  write_accessor output_accessor_;
};

template <typename T>
struct TileFunctor<SYCLDevice, T> {
  void operator()(const SYCLDevice& d, Tensor* out, const Tensor& in) {
    auto input_buffer = d.get_sycl_buffer(in.template flat<T>().data());
    auto output_buffer = d.get_sycl_buffer(out->template flat<T>().data());
    const int ndims = in.dims();
    const int64 nelem = out->NumElements();
    gtl::InlinedVector<int64, 8> in_strides = ComputeStride<int64>(in.shape());
    gtl::InlinedVector<int64, 8> out_strides =
        ComputeStride<int64>(out->shape());

    // Allocate a temporary [3 x ndims] SYCL buffer.
    // This holds the input strides, output strides and the input dimensions
    // needed for the kernel to compute the tile indices.
    cl::sycl::buffer<int64, 2> stride_buffer(cl::sycl::range<2>(3, ndims));
    auto stride_host = stride_buffer.template get_access<
        cl::sycl::access::mode::write>();
    for (int i = 0; i < ndims; ++i) {
      stride_host[cl::sycl::id<2>(0, i)] = in_strides[i];
      stride_host[cl::sycl::id<2>(1, i)] = out_strides[i];
      stride_host[cl::sycl::id<2>(2, i)] = in.shape().dim_size(i);
    }

    d.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access =
          input_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_access =
          output_buffer.template get_access<cl::sycl::access::mode::write>(cgh);
      auto stride_access =
          stride_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
      TileSYCL<T> functor(ndims, input_access, stride_access, output_access);

      cgh.parallel_for(cl::sycl::range<1>(nelem), functor);
    });
  }
};

#define DEFINE_TYPE(T)                        \
  template struct Tile<SYCLDevice, T, int32>; \
  template struct Tile<SYCLDevice, T, int64>;

TF_CALL_bool(DEFINE_TYPE);
TF_CALL_SYCL_NUMBER_TYPES(DEFINE_TYPE);
TF_CALL_uint8(DEFINE_TYPE);
TF_CALL_int32(DEFINE_TYPE);
TF_CALL_int16(DEFINE_TYPE);
TF_CALL_int64(DEFINE_TYPE);

#undef DEFINE_TYPE
#endif  // TENSORFLOW_USE_SYCL

}  // end namespace functor
}  // end namespace tensorflow
