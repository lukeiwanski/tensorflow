/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;

struct SYCL2DWindow {
  using Index = int;

  const Index rstart;
  const Index rend;
  const Index firstr;

  const Index cstart;
  const Index cend;
  const Index firstc;

  const Index feature;
  const Index batch;
};
struct SYCLConv2DParams {
  using Index = int;

  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCLConv2DParams(
      const Index channels, const Index features, const Index batch,
      const Index in_rows, const Index in_cols, const Index window_rows,
      const Index window_cols, const Index stride_rows, const Index stride_cols,
      const Index out_rows, const Index out_cols, const Index pad_rows,
      const Index pad_cols)
      :

        channels_{channels},
        features_{features},
        batch_{batch},
        in_rows_{in_rows},
        in_cols_{in_cols},
        window_rows_{window_rows},
        window_cols_{window_cols},
        stride_rows_{stride_rows},
        stride_cols_{stride_cols},
        out_rows_{out_rows},
        out_cols_{out_cols},
        pad_rows_{pad_rows},
        pad_cols_{pad_cols} {}

  /* The number of input channels. */
  const Index channels_;
  /* The number of output feature channels. */
  const Index features_;
  const Index batch_;

  const Index in_rows_;
  const Index in_cols_;

  const Index window_rows_;
  const Index window_cols_;

  const Index stride_rows_;
  const Index stride_cols_;

  const Index out_rows_;
  const Index out_cols_;

  const Index pad_rows_;
  const Index pad_cols_;

  /**
   * Get the index in the kernel tensor for a particular channel, row and
   * column.
   */
  inline TF_ATTRIBUTE_ALWAYS_INLINE Index kernel_index(const Index channel,
                                                       const Index feature,
                                                       const Index i,
                                                       const Index j) const {
    return (((i * window_cols_) + j) * channels_ + channel) * features_ +
           feature;
  }
  /**
   * Get the window in the input tensor which corresponds to the specified
   * output index.
   *
   * NOTE: The index types used here must be signed to ensure that the padding
   * is correctly calculated.
   */
  inline TF_ATTRIBUTE_ALWAYS_INLINE SYCL2DWindow
  input_window_from_output(const Index index) const {
    Index batch = index;
    const Index feature = batch % features_;
    batch /= features_;
    Index cstart = (batch % out_cols_) * stride_cols_ - pad_cols_;
    const Index cend = std::min(cstart + window_cols_, in_cols_);
    const Index firstc = cstart < 0 ? -cstart : 0;
    cstart = std::max(cstart, static_cast<Index>(0));
    batch /= out_cols_;
    Index rstart = (batch % out_rows_) * stride_rows_ - pad_rows_;
    const Index rend = std::min(rstart + window_rows_, in_rows_);
    const Index firstr = rstart < 0 ? -rstart : 0;
    rstart = std::max(rstart, static_cast<Index>(0));
    batch /= out_rows_;

    return {rstart, rend, firstr, cstart, cend, firstc, feature, batch};
  }
};
namespace functor {
/**
 * SYCL kernel for naive convolution computation.
 */
template <typename T>
struct Conv2DSYCL {
  using Index = int;
  using buffer_data = uint8_t;
  static constexpr auto read_mode = cl::sycl::access::mode::read;
  static constexpr auto write_mode = cl::sycl::access::mode::write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor =
      cl::sycl::accessor<buffer_data, 1, write_mode, global_access>;
  using read_accessor =
      cl::sycl::accessor<buffer_data, 1, read_mode, global_access>;

  inline TF_ATTRIBUTE_ALWAYS_INLINE Conv2DSYCL(Index n_elems,
                                               const SYCLConv2DParams& params,
                                               const read_accessor input,
                                               const read_accessor kernel,
                                               write_accessor output)
      : n_elems_{n_elems},
        p_{params},
        input_accessor_{input},
        kernel_accessor_{kernel},
        output_accessor_{output} {}

  inline TF_ATTRIBUTE_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    const T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
    const T* kernel_data = ConvertToActualTypeSycl(T, kernel_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_accessor_);
    const Index index = item.get_linear_id();

    if (index < n_elems_) {
      SYCL2DWindow w = p_.input_window_from_output(index);

      T out_val = T{0};
      const T* input_data_n =
          input_data + w.batch * p_.in_cols_ * p_.in_rows_ * p_.channels_;
      for (Index r = w.rstart, i = w.firstr; r < w.rend; ++r, ++i) {
        for (Index c = w.cstart, j = w.firstc; c < w.cend; ++c, ++j) {
          for (Index channel = 0; channel < p_.channels_; ++channel) {
            const Index idx = (r * p_.in_cols_ + c) * p_.channels_ + channel;
            const Index k_idx = p_.kernel_index(channel, w.feature, i, j);
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
}  // namespace functor

template <typename T>
struct LaunchConv2DSYCL {
  static constexpr auto read_mode = functor::Conv2DSYCL<T>::read_mode;
  static constexpr auto write_mode = functor::Conv2DSYCL<T>::write_mode;
  using Index = int;

  static void launch(OpKernelContext* context, Tensor* output,
                     const Tensor& tensor_in, const Tensor& filter,
                     const SYCLConv2DParams& params) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    const Index output_size = output->NumElements();
    const Index n_threads = output_size;

    auto input_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto filter_buffer =
        device.get_sycl_buffer(filter.template flat<T>().data());
    auto output_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());

    device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access = input_buffer.template get_access<read_mode>(cgh);
      auto filter_access = filter_buffer.template get_access<read_mode>(cgh);
      auto output_access = output_buffer.template get_access<write_mode>(cgh);

      functor::Conv2DSYCL<T> conv(output_size, params, input_access,
                                  filter_access, output_access);

      cgh.parallel_for(cl::sycl::range<1>(n_threads), conv);
    });
  }
};
template <typename T>
class LaunchDeepConvOp<SYCLDevice, T> {
 public:
  static bool Run(OpKernelContext* ctx, const Tensor& input,
                  const Tensor& filter, int batch, int input_rows,
                  int input_cols, int in_depth, int filter_rows,
                  int filter_cols, int pad_rows, int pad_cols, int out_rows,
                  int out_cols, int out_depth, int stride_rows, int stride_cols,
                  Tensor* output, TensorFormat data_format) {
    SYCLConv2DParams params{in_depth,    out_depth,   batch,       input_rows,
                            input_cols,  filter_rows, filter_cols, stride_rows,
                            stride_cols, out_rows,    out_cols,    pad_rows,
                            pad_cols};

    LaunchConv2DSYCL<T>::launch(ctx, output, input, filter, params);
    return true;
  }
};
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_
