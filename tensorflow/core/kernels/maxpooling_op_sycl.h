/*
 * Copyright 2017 John Lawson, Codeplay Software.
 * All rights reserved.
 */
#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_MAXPOOLING_OPS_SYCL_H_
#define TENSORFLOW_KERNELS_MAXPOOLING_OPS_SYCL_H_

#include <type_traits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;

/**
 * Round up the given to value to the next largest multiple of the given
 * multiplier. If the value is a multiple itself then the value will be
 * returned.
 */
template <typename IntegerType>
inline TF_ATTRIBUTE_ALWAYS_INLINE IntegerType
RoundUpToNearestMultiple(const IntegerType val, const IntegerType multiplier) {
  static_assert(
      std::is_integral<IntegerType>::value,
      "Rounding to nearest multiple is only valid for integer types.");
  const IntegerType diff = val % multiplier;
  return val + (multiplier - diff);
}
/**
 * MaxPool2D SYCL kernel. Expects the number of threads to be equal to the
 * number of elements in the output tensor.

 * For each output element, find the corresponding input window and run over
 * all values in the window to find the maximum value. This value is then
 * copied into that output element.
 */
template <typename T>
class MaxPool2DSYCL {
 public:
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, read_mode, global_access>;
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, write_mode, global_access>;

  MaxPool2DSYCL(const int n_threads, const PoolParameters& params,
                const read_accessor input_accessor,
                write_accessor output_accessor)
      : n_threads_{n_threads},
        p_{params},
        input_accessor_{input_accessor},
        output_accessor_{output_accessor} {}
  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

    const size_t num_elements = output_accessor_.get_size() / sizeof(T);

    int index = item.get(0);
    if (index < num_elements) {
      int n = index;
      int d = n % p_.depth_;
      n /= p_.depth_;
      int cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
      int cend = cl::sycl::min(cstart + p_.window_cols_, p_.in_cols_);
      cstart = cl::sycl::max(cstart, 0);
      n /= p_.out_cols_;
      int rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
      int rend = cl::sycl::min(rstart + p_.window_rows_, p_.in_rows_);
      rstart = cl::sycl::max(rstart, 0);
      n /= p_.out_rows_;
      T maxval = Eigen::NumTraits<T>::lowest();
      const T* input_data_n =
          input_data + n * p_.in_cols_ * p_.in_rows_ * p_.depth_;
      for (int r = rstart; r < rend; ++r) {
        for (int c = cstart; c < cend; ++c) {
          int idx = (r * p_.in_cols_ + c) * p_.depth_ + d;
          if (input_data_n[idx] > maxval) {
            maxval = input_data_n[idx];
          }
        }
      }
      output_data[index] = maxval;
    }
  }

 private:
  const int n_threads_;
  const SYCL2DPoolParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
template <typename T>
struct LaunchMaxPoolingOpSYCL {
  using Functor = MaxPool2DSYCL<T>;
  static constexpr auto read_mode = Functor::read_mode;
  static constexpr auto write_mode = Functor::write_mode;

  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const PoolParameters& params, Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();

    const int output_size = output->NumElements();
    const int workgroup_size = device.maxSyclThreadsPerBlock();
    const int n_threads = RoundUpToNearestMultiple(output_size, workgroup_size);

    auto input_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access = input_buffer.template get_access<read_mode>(cgh);
      auto output_access = output_buffer.template get_access<write_mode>(cgh);
      Functor max_pool(n_threads, params, input_access, output_access);

      cgh.parallel_for(cl::sycl::range<1>(n_threads), max_pool);
    });
  }
};
/**
 * MaxPoolGrad SYCL kernel. Expects the number of threads to be equal to the
 * number of elements in the output backprop tenor (i.e. the number of elements
 * in the input data tensor).
 *
 * For each output backprop element we compute the possible window of values in
 * the input backprop tensor which might contribute to this element. Then for
 * each error in this window, compute the corresponding input window which was
 * pooled into that element in the output. Walk through this input window to
 * determine whether the input value is the first maximum value, and so the
 * error should be propagated back to the corresponding backprop element.
 */
template <typename T>
class MaxPoolGradSYCL {
 public:
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, read_mode, global_access>;
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, write_mode, global_access>;

  MaxPoolGradSYCL(const int n_threads, const SYCL2DPoolParams& params,
                  const read_accessor input_data_accessor,
                  const read_accessor output_data_accessor,
                  const read_accessor input_backprop_accessor,
                  write_accessor output_backprop_accessor)
      : n_threads_{n_threads},
        p_{params},
        input_data_accessor_{input_data_accessor},
        output_data_accessor_{output_data_accessor},
        input_backprop_accessor_{input_backprop_accessor},
        output_backprop_accessor_{output_backprop_accessor} {}
  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_data_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_data_accessor_);
    T* input_backprop = ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);

    const size_t output_backprop_ret_size = output_backprop_accessor_.get_size() / sizeof(T);

    const int index = item.get(0);
    if (index < output_backprop_ret_size) {
      T output_value = static_cast<T>(0);
      int n = index;
      const int d = n % p_.depth_;
      n /= p_.depth_;
      const int c = (n % p_.in_cols_) + p_.pad_cols_;
      const int poolcstart = (c < p_.window_cols_)
                                 ? 0
                                 : (c - p_.window_cols_) / p_.stride_cols_ + 1;
      const int poolcend = cl::sycl::min(c / p_.stride_cols_ + 1, p_.out_cols_);
      n /= p_.in_cols_;
      const int r = (n % p_.in_rows_) + p_.pad_rows_;
      const int poolrstart = (r < p_.window_rows_)
                                 ? 0
                                 : (r - p_.window_rows_) / p_.stride_rows_ + 1;
      const int poolrend = cl::sycl::min(r / p_.stride_rows_ + 1, p_.out_rows_);
      n /= p_.in_rows_;
      const int index_no_n = index - n * p_.in_cols_ * p_.in_rows_ * p_.depth_;

      const T* input_data_n =
          input_data + n * p_.in_cols_ * p_.in_rows_ * p_.depth_;
      const T* output_data_n =
          output_data + n * p_.out_cols_ * p_.out_rows_ * p_.depth_;
      const T* input_backprop_n =
          input_backprop + n * p_.out_cols_ * p_.out_rows_ * p_.depth_;

      for (int poolr = poolrstart; poolr < poolrend; ++poolr) {
        int rstart = poolr * p_.stride_rows_ - p_.pad_rows_;
        const int rend = cl::sycl::min(rstart + p_.window_rows_, p_.in_rows_);
        rstart = cl::sycl::max(rstart, 0);

        for (int poolc = poolcstart; poolc < poolcend; ++poolc) {
          int cstart = poolc * p_.stride_cols_ - p_.pad_cols_;
          const int cend = cl::sycl::min(cstart + p_.window_cols_, p_.in_cols_);
          cstart = cl::sycl::max(cstart, 0);

          const int output_data_idx =
              (poolr * p_.out_cols_ + poolc) * p_.depth_ + d;
          bool should_continue = true;
          bool is_max = (input_data[index] == output_data_n[output_data_idx]);
          for (int win_r = rstart; win_r < rend && should_continue; ++win_r) {
            for (int win_c = cstart; win_c < cend && should_continue; ++win_c) {
              const int input_data_idx =
                  (win_r * p_.in_cols_ + win_c) * p_.depth_ + d;
              if (input_data_idx == index_no_n) {
                should_continue = false;
              } else if (input_data_n[input_data_idx] ==
                         output_data_n[output_data_idx]) {
                should_continue = false;
                is_max = false;
              }
            }
          }
          if (is_max) {
            output_value += input_backprop_n[output_data_idx];
          }
        }
      }
      output_backprop[index] = output_value;
    }
  }

 private:
  const int n_threads_;
  const SYCL2DPoolParams p_;

  const read_accessor input_data_accessor_;
  const read_accessor output_data_accessor_;
  const read_accessor input_backprop_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T>
struct LaunchMaxPoolingGradOpSYCL {
  using Functor = MaxPoolGradSYCL<T>;
  static constexpr auto read_mode = Functor::read_mode;
  static constexpr auto write_mode = Functor::write_mode;

  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const Tensor& tensor_out, const Tensor& out_backprop,
                     const SYCL2DPoolParams& params, Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();

    const int output_size = output->NumElements();
    const int workgroup_size = device.maxSyclThreadsPerBlock();
    const int n_threads = RoundUpToNearestMultiple(output_size, workgroup_size);

    auto input_data_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_data_buffer =
        device.get_sycl_buffer(tensor_out.template flat<T>().data());
    auto input_backprop_buffer =
        device.get_sycl_buffer(out_backprop.template flat<T>().data());
    auto output_backprop_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_data_access =
          input_data_buffer.template get_access<read_mode>(cgh);
      auto output_data_access =
          output_data_buffer.template get_access<read_mode>(cgh);
      auto input_backprop_access =
          input_backprop_buffer.template get_access<read_mode>(cgh);
      auto output_backprop_access =
          output_backprop_buffer.template get_access<write_mode>(cgh);
      Functor max_pool(n_threads, params, input_data_access, output_data_access,
                       input_backprop_access, output_backprop_access);

      cgh.parallel_for(cl::sycl::range<1>(n_threads), max_pool);
    });
  }
};
/**
 * MaxPoolGradGrad SYCL kernel. Expects the number of threads to be equal to
 * the number of elements in the output backprop tensor, i.e. the number of
 * elements in the output tensor.
 *
 * For each element in the output backprop tensor, find the corresponding input
 * window, and compare the input and output data to find the index of the
 * maximum value in the input tensor. This is then the index of the gradient to
 * pass through to the output backprop tensor.
 */
template <typename T>
class MaxPoolGradGradSYCL {
 public:
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, read_mode, global_access>;
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, write_mode, global_access>;

  MaxPoolGradGradSYCL(const int n_threads, const PoolParameters& params,
                      const read_accessor input_data_accessor,
                      const read_accessor output_data_accessor,
                      const read_accessor input_backprop_accessor,
                      write_accessor output_backprop_accessor)
      : n_threads_{n_threads},
        p_{params},
        input_data_accessor_{input_data_accessor},
        output_data_accessor_{output_data_accessor},
        input_backprop_accessor_{input_backprop_accessor},
        output_backprop_accessor_{output_backprop_accessor} {}
  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_data_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_data_accessor_);
    T* input_backprop = ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);

    const size_t output_data_ret_size = output_data_accessor_.get_size() / sizeof(T);

    int index = item.get(0);
    if (index < output_data_ret_size) {
      int n = index;
      int d = n % p_.depth_;
      n /= p_.depth_;
      int cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
      int cend = cl::sycl::min(cstart + p_.window_cols_, p_.in_cols_);
      cstart = cl::sycl::max(cstart, 0);
      n /= p_.out_cols_;
      int rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
      int rend = cl::sycl::min(rstart + p_.window_rows_, p_.in_rows_);
      rstart = cl::sycl::max(rstart, 0);
      n /= p_.out_rows_;
      int maxidx = -1;
      bool should_stop = false;
      const T* input_data_n =
          input_data + n * p_.in_cols_ * p_.in_rows_ * p_.depth_;
      for (int r = rstart; r < rend && !should_stop; ++r) {
        for (int c = cstart; c < cend && !should_stop; ++c) {
          int idx = (r * p_.in_cols_ + c) * p_.depth_ + d;
          if (output_data[index] == input_data_n[idx]) {
            maxidx = idx;
            should_stop = true;
          }
        }
      }
      if (maxidx != -1) {
        output_backprop[index] =
            input_backprop[n * p_.in_rows_ * p_.in_cols_ * p_.depth_ + maxidx];
      }
    }
  }

 private:
  const int n_threads_;
  const SYCL2DPoolParams p_;

  const read_accessor input_data_accessor_;
  const read_accessor output_data_accessor_;
  const read_accessor input_backprop_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T>
struct LaunchMaxPoolingGradGradOpSYCL {
  using Functor = MaxPoolGradGradSYCL<T>;
  static constexpr auto read_mode = Functor::read_mode;
  static constexpr auto write_mode = Functor::write_mode;

  static void launch(OpKernelContext* context, const PoolParameters& params,
                     const Tensor& tensor_in, const Tensor& tensor_out,
                     const Tensor& out_backprop, Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();

    const int output_size = output->NumElements();
    const int workgroup_size = device.maxSyclThreadsPerBlock();
    const int n_threads = RoundUpToNearestMultiple(output_size, workgroup_size);

    auto input_data_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_data_buffer =
        device.get_sycl_buffer(tensor_out.template flat<T>().data());
    auto input_backprop_buffer =
        device.get_sycl_buffer(out_backprop.template flat<T>().data());
    auto output_backprop_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_data_access =
          input_data_buffer.template get_access<read_mode>(cgh);
      auto output_data_access =
          output_data_buffer.template get_access<read_mode>(cgh);
      auto input_backprop_access =
          input_backprop_buffer.template get_access<read_mode>(cgh);
      auto output_backprop_access =
          output_backprop_buffer.template get_access<write_mode>(cgh);
      Functor maxpoolgradgrad(n_threads, params, input_data_access,
                              output_data_access, input_backprop_access,
                              output_backprop_access);

      cgh.parallel_for(cl::sycl::range<1>(n_threads), maxpoolgradgrad);
    });
  }
};
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_MAXPOOLING_OPS_SYCL_H_
