#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_NAIVE_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_NAIVE_SYCL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_ops_sycl_common.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
namespace functor {
/**
 * SYCL kernel for naive convolution computation.
 */
template <typename T, ConvType CType>
struct Conv2DNaiveSYCL;
template <typename T>
struct Conv2DNaiveSYCL<T, ConvType::Forward> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE Conv2DNaiveSYCL(
      Index n_elems, const SYCLConv2DParams& params, const read_accessor input,
      const read_accessor kernel, write_accessor output)
      : n_elems_{n_elems},
        p_{params},
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const Index index = item.get(0);

    if (index < n_elems_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index feature = index % p_.features_;
      const Index tile_idx = index / p_.features_;
      const SYCL2DWindow w = p_.input_window_from_output(tile_idx);

      T out_val = static_cast<T>(0);
      const T* input_data_n =
          input_data + w.batch * p_.in_cols_ * p_.in_rows_ * p_.channels_;
      for (Index r = w.rstart, i = 0; r < w.rend; ++r, ++i) {
        if (r >= 0) {
          for (Index c = w.cstart, j = 0; c < w.cend; ++c, ++j) {
            if (c >= 0) {
              for (Index channel = 0; channel < p_.channels_; ++channel) {
                const Index idx =
                    (r * p_.in_cols_ + c) * p_.channels_ + channel;
                const Index k_idx = p_.kernel_index(channel, feature, i, j);
                out_val += input_data_n[idx] * kernel_data[k_idx];
              }
            }
          }
        }
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
template <typename T>
struct Conv2DNaiveSYCL<T, ConvType::InputBackprop> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE Conv2DNaiveSYCL(
      Index n_elems, const SYCLConv2DParams& params, const read_accessor input,
      const read_accessor kernel, write_accessor output)
      : n_elems_{n_elems},
        p_{params},
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const Index index = item.get(0);
    if (index < n_elems_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);

      const Index feature = index % p_.features_;
      const Index tile_idx = index / p_.features_;
      const SYCL2DWindow w = p_.output_window_from_input_no_dilation(tile_idx);

      T out_val = static_cast<T>(0);
      const T* input_data_n =
          input_data + w.batch * p_.out_cols_ * p_.out_rows_ * p_.channels_;
      for (Index r = w.rstart, i = w.firstr; r < w.rend;
           ++r, i += p_.stride_rows_) {
        for (Index c = w.cstart, j = w.firstc; c < w.cend;
             ++c, j += p_.stride_cols_) {
          for (Index channel = 0; channel < p_.channels_; ++channel) {
            const Index idx = (r * p_.out_cols_ + c) * p_.channels_ + channel;
            const Index k_idx = p_.backprop_index(feature, channel, i, j);
            out_val += input_data_n[idx] * kernel_data[k_idx];
          }
        }
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
/*
 * The main difference between the two backprop kernels is the way strides are
 * handled. In the filter backprop the input is strided and the kernel is not
 * whereas in the input backprop this is the other way around.
 */
template <typename T>
struct Conv2DNaiveSYCL<T, ConvType::FilterBackprop> {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE Conv2DNaiveSYCL(
      Index n_elems, const SYCLConv2DParams& params, const read_accessor input,
      const read_accessor kernel, write_accessor output)
      : n_elems_{n_elems},
        p_{params},
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const Index index = item.get(0);
    if (index < n_elems_) {
      const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
      const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
      T* output_data = ConvertToActualTypeSycl(T, output_accessor_);
      const SYCL2DKernelWindow w = p_.kernel_window_from_output(index);

      T out_val = static_cast<T>(0);
      const T* input_data_n = input_data;
      for (Index b = 0; b < p_.batch_; b++) {
        for (Index r = w.rstart, i = 0; r < w.rend; ++i, r += p_.stride_rows_) {
          if (r >= 0) {
            for (Index c = w.cstart, j = 0; c < w.cend;
                 ++j, c += p_.stride_cols_) {
              if (c >= 0) {
                const Index idx =
                    (r * p_.in_cols_ + c) * p_.channels_ + w.channel;
                const Index k_idx = p_.filter_kernel_index(b, i, j, w.feature);
                out_val += input_data_n[idx] * kernel_data[k_idx];
              }
            }
          }
        }
        input_data_n += p_.in_cols_ * p_.in_rows_ * p_.channels_;
      }
      output_data[index] = out_val;
    }
  }

 private:
  const Index n_elems_;
  const SYCLConv2DParams p_;
  const read_accessor input_accessor_;
  const read_accessor kernel_accessor_;
  write_accessor output_accessor_;
};
template <ConvType CType>
inline SYCLConv2DParams get_kernel_params(SYCLConv2DParams params) {
  return params;
}
template <>
inline SYCLConv2DParams get_kernel_params<ConvType::InputBackprop>(
    SYCLConv2DParams params) {
  std::swap(params.channels_, params.features_);
  return params;
}
template <>
inline SYCLConv2DParams get_kernel_params<ConvType::FilterBackprop>(
    SYCLConv2DParams params) {
  // Map the input dimensions to those expected in the convolution kernel.
  const auto window_rows =
      params.out_rows_ * params.stride_rows_ - (params.stride_rows_ - 1);
  const auto window_cols =
      params.out_cols_ * params.stride_cols_ - (params.stride_cols_ - 1);
  params.out_rows_ = params.window_rows_;
  params.out_cols_ = params.window_cols_;
  params.window_rows_ = window_rows;
  params.window_cols_ = window_cols;
  return params;
}
template <ConvType CType>
inline size_t get_output_size(SYCLConv2DParams const& params);
template <>
inline size_t get_output_size<ConvType::Forward>(
    SYCLConv2DParams const& params) {
  return params.batch_ * params.out_rows_ * params.out_cols_ * params.features_;
}
template <>
inline size_t get_output_size<ConvType::InputBackprop>(
    SYCLConv2DParams const& params) {
  return params.batch_ * params.in_rows_ * params.in_cols_ * params.channels_;
}
template <>
inline size_t get_output_size<ConvType::FilterBackprop>(
    SYCLConv2DParams const& params) {
  return params.window_rows_ * params.window_cols_ * params.channels_ *
         params.features_;
}

}  // namespace functor

template <typename T, ConvType CType>
struct LaunchConv2DKernel {
  using Functor = functor::Conv2DNaiveSYCL<T, CType>;
  static constexpr auto read_mode = Functor::read_mode;
  static constexpr auto write_mode = Functor::write_mode;
  using Index = int;

  static void launch(Eigen::SyclDevice const& device, T* const output,
                     T const* const input, T const* const filter,
                     SYCLConv2DParams const& params) {
    const Index output_size = functor::get_output_size<CType>(params);
    const Index workgroup_size = device.maxSyclThreadsPerBlock();
    const Index n_threads =
        RoundUpToNearestMultiple(output_size, workgroup_size);

    auto input_buffer = device.get_sycl_buffer(input);
    auto filter_buffer = device.get_sycl_buffer(filter);
    auto output_buffer = device.get_sycl_buffer(output);
    auto kernel_params = functor::get_kernel_params<CType>(params);

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access = input_buffer.template get_access<read_mode>(cgh);
      auto filter_access = filter_buffer.template get_access<read_mode>(cgh);
      auto output_access = output_buffer.template get_access<write_mode>(cgh);

      Functor conv(output_size, kernel_params, input_access, filter_access,
                   output_access);

      cgh.parallel_for(cl::sycl::range<1>(n_threads), conv);
    });
  }
};
template <typename T, typename backend_type, ConvType CType>
struct Launcher<T, backend_type, algorithm::direct, CType> final
    : public LaunchConv2DKernel<T, CType> {};

template <typename T>
struct LaunchConv2DSYCL final
    : public LaunchConv2DKernel<T, ConvType::Forward> {};

template <typename T>
struct LaunchConv2DBackpropInputSYCL final
    : public LaunchConv2DKernel<T, ConvType::InputBackprop> {};

template <typename T>
struct LaunchConv2DBackpropFilterSYCL final
    : public LaunchConv2DKernel<T, ConvType::FilterBackprop> {};

}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_NAIVE_SYCL_H_
