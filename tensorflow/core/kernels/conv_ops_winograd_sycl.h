#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_H_

#include "tensorflow/core/kernels/conv_ops_winograd_sycl_kernels.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
namespace winograd {
struct TileInfo {
  int rows;
  int cols;
  int number;
};
template <ConvType CType>
inline TileInfo get_tile_info(SYCLConv2DParams const& params, int out_tile_rows,
                              int out_tile_cols) {
  const int n_tile_rows =
      RoundRatioUpAboveZero(params.out_rows_, out_tile_rows);
  const int n_tile_cols =
      RoundRatioUpAboveZero(params.out_cols_, out_tile_cols);
  const int n_tiles = n_tile_rows * n_tile_cols;
  const TileInfo result{n_tile_rows, n_tile_cols, n_tiles};
  return result;
}
template <>
inline TileInfo get_tile_info<ConvType::FilterBackprop>(
    SYCLConv2DParams const& params, int out_tile_rows, int out_tile_cols) {
  const int n_tile_rows =
      RoundRatioUpAboveZero(params.window_rows_, out_tile_rows);
  const int n_tile_cols =
      RoundRatioUpAboveZero(params.window_cols_, out_tile_cols);
  const int n_tiles = n_tile_rows * n_tile_cols;
  const TileInfo result{n_tile_rows, n_tile_cols, n_tiles};
  return result;
}
template <ConvType CType>
inline SYCLConv2DParams get_params(SYCLConv2DParams params) {
  return params;
}
template <>
inline SYCLConv2DParams get_params<ConvType::InputBackprop>(
    SYCLConv2DParams params) {
  std::swap(params.channels_, params.features_);
  std::swap(params.in_rows_, params.out_rows_);
  std::swap(params.in_cols_, params.out_cols_);
  // We need to change the padding from input padding to output padding for
  // the winograd matmul kernel. pad_out = filt_size - 1 - pad_in
  params.pad_rows_ = params.window_rows_ - 1 - params.pad_rows_;
  params.pad_cols_ = params.window_cols_ - 1 - params.pad_cols_;
  return params;
}
template <>
inline SYCLConv2DParams get_params<ConvType::FilterBackprop>(
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
struct AllocInfo {
  size_t alloc_limit;
  size_t images_per_alloc;
  size_t n_input_transforms;
  size_t last_batch_size;
  bool alloc_warning;
};
template <ConvType CType>
inline AllocInfo get_alloc_info(cl::sycl::queue& queue,
                                cl::sycl::device const& device,
                                size_t const batch_size,
                                size_t const alloc_size_per_image,
                                bool try_wait_if_alloc_fails = true) {
  size_t alloc_limit =
      device.get_info<cl::sycl::info::device::max_mem_alloc_size>() / 4;
  bool alloc_warning = false;
  if (TF_PREDICT_FALSE(alloc_size_per_image > alloc_limit)) {
    if (try_wait_if_alloc_fails) {
      VLOG(2) << "Winograd requires a temporary buffer that is too large to "
                 "allocate. Waiting for the device to clear its queue before "
                 "trying again.";
      queue.wait_and_throw();
      return get_alloc_info<CType>(queue, device, batch_size,
                                   alloc_size_per_image, false);
    } else {
      VLOG(1) << "The temporary buffer required by Winograd for a single "
                 "image is too large to be allocated on the device. This "
                 "is likely to cause a CL_MEM_OBJECT_ALLOCATION_FAILURE "
                 "OpenCL error.";
      VLOG(2) << "buffer size per image: " << alloc_size_per_image
              << ", device allocation limit: " << alloc_limit;
      alloc_limit = alloc_size_per_image + 1;
      alloc_warning = true;
    }
  }
  // The number of images per alloc is bounded above by the total number of
  // images in a batch
  size_t images_per_alloc =
      std::min<size_t>(batch_size, alloc_limit / alloc_size_per_image);

  const size_t n_input_transforms =
      RoundRatioUpAboveZero<size_t>(batch_size, images_per_alloc);
  images_per_alloc =
      RoundRatioUpAboveZero<size_t>(batch_size, n_input_transforms);
  assert(images_per_alloc * alloc_size_per_image < alloc_limit);
  const size_t last_batch_size =
      batch_size - images_per_alloc * (n_input_transforms - 1);
  assert(last_batch_size > 0);
  assert(last_batch_size <= batch_size);
  const AllocInfo result{alloc_limit, images_per_alloc, n_input_transforms,
                         last_batch_size, alloc_warning};
  return result;
}
struct Offsets {
  int in;
  int out;
};
template <ConvType CType>
inline Offsets calculate_offsets(int i, int images_per_alloc,
                                 SYCLConv2DParams const& params) {
  const int in_offset = i * images_per_alloc * params.in_rows_ *
                        params.in_cols_ * params.channels_;
  const int out_offset = i * images_per_alloc * params.out_rows_ *
                         params.out_cols_ * params.features_;
  Offsets result{in_offset, out_offset};
  return result;
}
template <>
inline Offsets calculate_offsets<ConvType::FilterBackprop>(
    int i, int images_per_alloc, SYCLConv2DParams const& params) {
  const int in_offset = i * images_per_alloc * params.in_rows_ *
                        params.in_cols_ * params.channels_;
  const int out_offset = i * images_per_alloc * params.window_rows_ *
                         params.window_cols_ * params.features_;
  Offsets result{in_offset, out_offset};
  return result;
}
}  // namespace winograd
template <typename T, int channel_vector, int M, int N, int R, int S,
          ConvType CType>
struct LaunchMatmulWinograd {
  using Index = int;
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  static constexpr bool trans_input = false;
  static constexpr bool trans_filter = true;//(CType == ConvType::Forward);
  using InputTransform =
      winograd::ExtractInputTiles<T, channel_vector, M, N, R, S, CType>;
  using FilterTransform = winograd::ExtractKernelTiles<T, M, N, R, S, CType>;
  using OutputTransform =
      winograd::ExtractOutputTiles<T, M, N, R, S, CType, false>;

  static bool launch(Eigen::SyclDevice const& device, T* const output,
                     T const* const input, T const* const filter,
                     SYCLConv2DParams const& params) {
    const winograd::TileInfo tile_info =
        winograd::get_tile_info<CType>(params, M, N);
    cl::sycl::queue sycl_queue = device.sycl_queue();
    cl::sycl::device sycl_device = sycl_queue.get_device();

    size_t const fil_transform_bytes =
        A * B * params.channels_ * params.features_ * sizeof(T);
    const size_t alloc_limit =
        sycl_device.get_info<cl::sycl::info::device::max_mem_alloc_size>() / 4;
    if (TF_PREDICT_FALSE(fil_transform_bytes > alloc_limit)) {
      VLOG(1) << "The temporary buffer required by Winograd for the "
                 "filter transform is too large to be allocated on the "
                 "device. This is likely to cause a "
                 "CL_MEM_OBJECT_ALLOCATION_FAILURE OpenCL error.";
      VLOG(2) << "buffer size per image: " << fil_transform_bytes
              << ", device allocation limit: " << alloc_limit;
      return false;
    }
    T* const fil_transform =
        static_cast<T*>(device.allocate(fil_transform_bytes));
    const Index fil_transform_items = params.features_ * params.channels_;
    sycl_conv::launch_transform<FilterTransform>(
        device, filter, fil_transform, fil_transform_items, params, 0);

    size_t const in_alloc_size_per_image =
        A * B * tile_info.number * params.channels_ * sizeof(T);
    size_t const inter_alloc_size_per_image =
        A * B * tile_info.number * params.features_ * sizeof(T);
    size_t const max_alloc_size_per_image =
        std::max(in_alloc_size_per_image, inter_alloc_size_per_image);
    winograd::AllocInfo const alloc_info = winograd::get_alloc_info<CType>(
        sycl_queue, sycl_device, params.batch_, max_alloc_size_per_image);
    size_t images_per_alloc = alloc_info.images_per_alloc;
    if (alloc_info.alloc_warning) {
      return false;
    }

    size_t const in_transform_bytes =
        in_alloc_size_per_image * images_per_alloc;
    T* const in_transform =
        static_cast<T*>(device.allocate(in_transform_bytes));
    size_t const inter_bytes = inter_alloc_size_per_image * images_per_alloc;
    T* const intermediate = static_cast<T*>(device.allocate(inter_bytes));
    cl::sycl::event last_event;

    for (int i = 0; i < alloc_info.n_input_transforms; ++i) {
      winograd::Offsets offset =
          winograd::calculate_offsets<CType>(i, images_per_alloc, params);
      assert(i == 0 || offset.in > 0);
      assert(i == 0 || offset.out > 0);
      if (i == alloc_info.n_input_transforms - 1) {
        images_per_alloc = alloc_info.last_batch_size;
      }
      SYCLConv2DParams kernel_params{params};
      kernel_params.batch_ = images_per_alloc;

      Index const in_transform_items = tile_info.number * kernel_params.batch_ *
                                       kernel_params.channels_ / channel_vector;
      sycl_conv::launch_transform<InputTransform>(
          device, input, in_transform, in_transform_items, kernel_params,
          offset.in, tile_info.number * kernel_params.batch_, tile_info.rows,
          tile_info.cols);

      sycl_conv::launch_batch_matmul<trans_input, trans_filter>(
          device, in_transform, fil_transform, intermediate, A * B,
          tile_info.number * kernel_params.batch_, kernel_params.channels_,
          kernel_params.features_);

      Index const n_out_items =
          tile_info.number * kernel_params.batch_ * kernel_params.features_;
      last_event = sycl_conv::launch_transform<OutputTransform>(
          device, intermediate, output, n_out_items, kernel_params, offset.out,
          tile_info.number * kernel_params.batch_, tile_info.rows,
          tile_info.cols);
    }
    device.deallocate(fil_transform);
    device.deallocate(in_transform);
    device.deallocate(intermediate);
    last_event.wait();
    return true;
  }
};
template <typename T, int channel_vector, int M, int N, int R, int S>
struct LaunchMatmulWinograd<T, channel_vector, M, N, R, S,
                            ConvType::FilterBackprop> {
  using Index = int;
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  static constexpr auto CType = ConvType::FilterBackprop;
  using InputTransform =
      winograd::ExtractInputTiles<T, channel_vector, M, N, R, S, CType>;
  using FilterTransform = winograd::ExtractKernelTiles<T, M, N, R, S, CType>;
  using OutputTransform =
      winograd::ExtractOutputTiles<T, M, N, R, S, CType, false>;
  using OutputTransformAccumulate =
      winograd::ExtractOutputTiles<T, M, N, R, S, CType, true>;

  static bool launch(Eigen::SyclDevice const& device, T* const output,
                     T const* const input, T const* const filter,
                     SYCLConv2DParams const& params) {
    const winograd::TileInfo tile_info =
        winograd::get_tile_info<CType>(params, R, S);

    cl::sycl::queue& sycl_queue = device.sycl_queue();
    cl::sycl::device sycl_device = sycl_queue.get_device();

    size_t const inter_bytes =
        A * B * params.channels_ * params.features_ * sizeof(T);
    size_t const alloc_limit =
        sycl_device.get_info<cl::sycl::info::device::max_mem_alloc_size>() / 4;
    if (TF_PREDICT_FALSE(inter_bytes > alloc_limit)) {
      VLOG(1) << "The temporary buffer required by Winograd for the "
                 "intermediate tensor is too large to be allocated on the "
                 "device. This is likely to cause a "
                 "CL_MEM_OBJECT_ALLOCATION_FAILURE OpenCL error.";
      VLOG(2) << "buffer size per image: " << inter_bytes
              << ", device allocation limit: " << alloc_limit;
      return false;
    }
    size_t const in_alloc_size_per_image =
        A * B * tile_info.number * params.channels_ * sizeof(T);
    size_t const fil_alloc_size_per_image =
        A * B * tile_info.number * params.features_ * sizeof(T);
    size_t const max_alloc_size_per_image =
        std::max(in_alloc_size_per_image, fil_alloc_size_per_image);
    const winograd::AllocInfo alloc_info = winograd::get_alloc_info<CType>(
        sycl_queue, sycl_device, params.batch_, max_alloc_size_per_image);
    size_t images_per_alloc = alloc_info.images_per_alloc;
    if (alloc_info.alloc_warning) {
      return false;
    }

    size_t const in_transform_bytes =
        in_alloc_size_per_image * images_per_alloc;
    size_t const fil_transform_bytes =
        fil_alloc_size_per_image * images_per_alloc;

    T* const in_transform =
        static_cast<T*>(device.allocate(in_transform_bytes));
    T* const fil_transform =
        static_cast<T*>(device.allocate(fil_transform_bytes));
    T* const intermediate = static_cast<T*>(device.allocate(inter_bytes));
    cl::sycl::event last_event;

    for (int i = 0; i < alloc_info.n_input_transforms; ++i) {
      winograd::Offsets offset =
          winograd::calculate_offsets<CType>(i, images_per_alloc, params);
      assert(i == 0 || offset.in > 0);
      assert(i == 0 || offset.out > 0);
      if (i == alloc_info.n_input_transforms - 1) {
        images_per_alloc = alloc_info.last_batch_size;
      }
      SYCLConv2DParams kernel_params{params};
      kernel_params.batch_ = images_per_alloc;

      const Index in_transform_items = tile_info.number * kernel_params.batch_ *
                                       kernel_params.channels_ / channel_vector;
      sycl_conv::launch_transform<InputTransform>(
          device, input, in_transform, in_transform_items, kernel_params,
          offset.in, tile_info.number * kernel_params.batch_, tile_info.rows,
          tile_info.cols);

      const Index fil_transform_items =
          kernel_params.features_ * tile_info.number * kernel_params.batch_;
      sycl_conv::launch_transform<FilterTransform>(
          device, filter, fil_transform, fil_transform_items, kernel_params,
          offset.out, tile_info.number * kernel_params.batch_, tile_info.rows,
          tile_info.cols);

      sycl_conv::launch_batch_matmul<true, false>(
          device, in_transform, fil_transform, intermediate, A * B,
          kernel_params.channels_, tile_info.number * kernel_params.batch_,
          kernel_params.features_);

      const Index out_transform_items =
          kernel_params.channels_ * kernel_params.features_;
      if (i == 0) {
        last_event = sycl_conv::launch_transform<OutputTransform>(
            device, intermediate, output, out_transform_items, kernel_params,
            tile_info.number * kernel_params.batch_);
      } else {
        last_event = sycl_conv::launch_transform<OutputTransformAccumulate>(
            device, intermediate, output, out_transform_items, kernel_params,
            tile_info.number * kernel_params.batch_);
      }
    }
    device.deallocate(fil_transform);
    device.deallocate(in_transform);
    device.deallocate(intermediate);
    last_event.wait();
    return true;
  }
};
}  // namespace tensorflow
#include "tensorflow/core/kernels/conv_ops_winograd_sycl_impl.h"
namespace tensorflow {
template <typename T, typename backend_type, int M, int N, int R, int S,
          ConvType CType>
struct WinogradVectorLauncher {
  static bool launch(backend_type const& backend, T* const output,
                     T const* const input, T const* const filter,
                     SYCLConv2DParams params) {
    params = winograd::get_params<CType>(params);
    if (params.channels_ % 4 == 0) {
      return LaunchMatmulWinograd<T, 4, M, N, R, S, CType>::launch(
          backend, output, input, filter, params);
    } else if (params.channels_ % 2 == 0) {
      return LaunchMatmulWinograd<T, 2, M, N, R, S, CType>::launch(
          backend, output, input, filter, params);
    } else {
      return LaunchMatmulWinograd<T, 1, M, N, R, S, CType>::launch(
          backend, output, input, filter, params);
    }
  }
};
template <typename T, typename backend_type, ConvType CType>
struct Launcher<T, backend_type, algorithm::winograd_3x3, CType> final
    : public WinogradVectorLauncher<T, backend_type, 2, 2, 3, 3, CType> {};
template <typename T, typename backend_type, ConvType CType>
struct Launcher<T, backend_type, algorithm::winograd_3x1, CType> final
    : public WinogradVectorLauncher<T, backend_type, 2, 1, 3, 1, CType> {};
template <typename T, typename backend_type, ConvType CType>
struct Launcher<T, backend_type, algorithm::winograd_1x3, CType> final
    : public WinogradVectorLauncher<T, backend_type, 1, 2, 1, 3, CType> {};

template <typename T, typename backend_type>
struct Launcher<T, backend_type, algorithm::winograd_3x3,
                ConvType::FilterBackprop>
    final : public WinogradVectorLauncher<T, backend_type, 3, 3, 2, 2,
                                          ConvType::FilterBackprop> {};
template <typename T, typename backend_type>
struct Launcher<T, backend_type, algorithm::winograd_3x1,
                ConvType::FilterBackprop>
    final : public WinogradVectorLauncher<T, backend_type, 3, 1, 2, 1,
                                          ConvType::FilterBackprop> {};
template <typename T, typename backend_type>
struct Launcher<T, backend_type, algorithm::winograd_1x3,
                ConvType::FilterBackprop>
    final : public WinogradVectorLauncher<T, backend_type, 1, 3, 1, 2,
                                          ConvType::FilterBackprop> {};
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_H_
