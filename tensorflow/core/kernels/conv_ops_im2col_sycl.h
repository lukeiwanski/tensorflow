#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/kernels/conv_ops_im2col_sycl_kernels.h"
#include "tensorflow/core/kernels/conv_ops_sycl_common.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace im2col {
struct TileInfo {
  int number;
  int size;
};
template <ConvType CType>
inline TileInfo get_tile_info(SYCLConv2DParams const& params);
template <>
inline TileInfo get_tile_info<ConvType::Forward>(
    SYCLConv2DParams const& params) {
  const int n_tiles = params.out_rows_ * params.out_cols_;
  const int tile_size =
      params.window_rows_ * params.window_cols_ * params.channels_;
  const TileInfo result{n_tiles, tile_size};
  return result;
}
template <>
inline TileInfo get_tile_info<ConvType::InputBackprop>(
    SYCLConv2DParams const& params) {
  const int n_tiles = params.in_rows_ * params.in_cols_;
  const int tile_size =
      params.window_rows_ * params.window_cols_ * params.features_;
  const TileInfo result{n_tiles, tile_size};
  return result;
}
template <>
inline TileInfo get_tile_info<ConvType::FilterBackprop>(
    SYCLConv2DParams const& params) {
  const int n_tiles =
      params.window_rows_ * params.window_cols_ * params.channels_;
  const int tile_size = params.out_rows_ * params.out_cols_;
  const TileInfo result{n_tiles, tile_size};
  return result;
}
struct AllocInfo {
  size_t alloc_limit;
  size_t images_per_alloc;
  size_t n_input_transforms;
  size_t last_batch_size;
  bool alloc_warning;
};
template <ConvType CType>
inline AllocInfo get_alloc_info(cl::sycl::queue queue,
                                cl::sycl::device const& device,
                                size_t const batch_size,
                                size_t const alloc_size_per_image,
                                bool try_wait_if_alloc_fails = true) {
  size_t alloc_limit =
      device.get_info<cl::sycl::info::device::max_mem_alloc_size>() / 4;
  bool alloc_warning = false;
  if (TF_PREDICT_FALSE(alloc_size_per_image > alloc_limit)) {
    if (try_wait_if_alloc_fails) {
      VLOG(2) << "Im2col requires a temporary buffer that is too large to "
                 "allocate. Waiting for the device to clear its queue before "
                 "trying again.";
      queue.wait_and_throw();
      return get_alloc_info<CType>(queue, device, batch_size,
                                   alloc_size_per_image, false);
    } else {
      VLOG(1) << "The temporary buffer required by im2col for a single "
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
/**
 * For the forward and filter backprop convolutions, the filter does not need to
 * be transformed, so we just reuse the original filter. As we don't allocate a
 * new tensor we don't deallocate after computing the convolution.
 */
template <typename T, ConvType CType>
struct FilterTransformAllocator {
  static T const* get_transform(Eigen::SyclDevice const& /*device*/,
                                T const* const filter,
                                SYCLConv2DParams const& /*params*/) {
    return filter;
  }
  static void deallocate(Eigen::SyclDevice const& /*device*/,
                         T const* const /*filter_transform*/) {}
};
/**
 * For the input backprop pass the filter has to be transformed, so we need to
 * allocate a tensor for the transform and compute the transform. The
 * transformed filter needs to be deallocated after computing the convolution.
 */
template <typename T>
struct FilterTransformAllocator<T, ConvType::InputBackprop> {
  using Index = int;
  static constexpr auto CType = ConvType::InputBackprop;
  using FilterTransform = im2col::ExtractKernelTiles<T, CType>;
  static T const* get_transform(Eigen::SyclDevice const& device,
                                T const* const filter,
                                SYCLConv2DParams const& params) {
    const size_t filter_size = params.window_rows_ * params.window_cols_ *
                               params.channels_ * params.features_;
    const size_t filter_size_bytes = filter_size * sizeof(T);
    T* const filter_transform =
        static_cast<T*>(device.allocate(filter_size_bytes));
    const Index fil_transform_items = params.window_rows_ *
                                      params.window_cols_ * params.channels_ *
                                      params.features_;
    sycl_conv::launch_transform<FilterTransform>(
        device, filter, filter_transform, fil_transform_items, params, 0);
    return filter_transform;
  }
  static void deallocate(Eigen::SyclDevice const& device,
                         T const* const filter_transform) {
    // Need to strip the const from the pointer type to allow the device to
    // deallocate it. The alternative to this const_cast would be to remove the
    // const from the pointer types used everywhere else, but it is useful to
    // have the extra const checking.
    T* filter = const_cast<T*>(filter_transform);
    device.deallocate(filter);
  }
};
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
inline Offsets calculate_offsets<ConvType::InputBackprop>(
    int i, int images_per_alloc, SYCLConv2DParams const& params) {
  const int in_offset = i * images_per_alloc * params.out_rows_ *
                        params.out_cols_ * params.features_;
  const int out_offset = i * images_per_alloc * params.in_rows_ *
                         params.in_cols_ * params.channels_;
  Offsets result{in_offset, out_offset};
  return result;
}
template <>
inline Offsets calculate_offsets<ConvType::FilterBackprop>(
    int i, int images_per_alloc, SYCLConv2DParams const& params) {
  const int in_offset = i * images_per_alloc * params.in_rows_ *
                        params.in_cols_ * params.channels_;
  const int out_offset = i * images_per_alloc * params.out_rows_ *
                         params.out_cols_ * params.features_;
  Offsets result{in_offset, out_offset};
  return result;
}
template <typename T, int vector_width, ConvType CType>
struct LaunchVectorTransform {
  using Index = int;
  using InputTransform = im2col::ExtractInputTiles<T, vector_width, CType>;
  static cl::sycl::event launch(Eigen::SyclDevice const& device,
                                T const* const input, Index const in_offset,
                                T* const transform,
                                const SYCLConv2DParams& params,
                                Index const tile_size) {
    const Index in_transform_items = params.batch_ * params.in_rows_ *
                                     params.in_cols_ * params.channels_ /
                                     vector_width;
    return sycl_conv::launch_transform<InputTransform>(
        device, input, transform, in_transform_items, params, in_offset,
        tile_size);
  }
};
template <typename T, int vector_width>
struct LaunchVectorTransform<T, vector_width, ConvType::InputBackprop> {
  using Index = int;
  static constexpr auto CType = ConvType::InputBackprop;
  using InputTransform = im2col::ExtractInputTiles<T, vector_width, CType>;
  static cl::sycl::event launch(Eigen::SyclDevice const& device,
                                T const* const input, Index const in_offset,
                                T* const transform,
                                const SYCLConv2DParams& params,
                                Index const tile_size) {
    const Index in_transform_items = params.batch_ * params.out_rows_ *
                                     params.out_cols_ * params.features_ /
                                     vector_width;
    return sycl_conv::launch_transform<InputTransform>(
        device, input, transform, in_transform_items, params, in_offset,
        tile_size);
  }
};
template <typename T, ConvType ConvType>
struct LaunchIm2colTransform;
template <typename T>
struct LaunchIm2colTransform<T, ConvType::Forward> {
  using Index = int;
  static constexpr auto CType = ConvType::Forward;
  static cl::sycl::event launch(Eigen::SyclDevice const& device,
                                T const* const input, Index const in_offset,
                                T* const transform,
                                const SYCLConv2DParams& params,
                                Index const tile_size) {
    if (params.channels_ % 4 == 0) {
      return LaunchVectorTransform<T, 4, CType>::launch(
          device, input, in_offset, transform, params, tile_size);
    } else if (params.channels_ % 2 == 0) {
      return LaunchVectorTransform<T, 2, CType>::launch(
          device, input, in_offset, transform, params, tile_size);
    } else {
      return LaunchVectorTransform<T, 1, CType>::launch(
          device, input, in_offset, transform, params, tile_size);
    }
  }
};
template <typename T>
struct LaunchIm2colTransform<T, ConvType::InputBackprop> {
  using Index = int;
  static constexpr auto CType = ConvType::InputBackprop;
  static cl::sycl::event launch(Eigen::SyclDevice const& device,
                                T const* const input, Index const in_offset,
                                T* const transform,
                                const SYCLConv2DParams& params,
                                Index const tile_size) {
    if (params.features_ % 4 == 0) {
      return LaunchVectorTransform<T, 4, CType>::launch(
          device, input, in_offset, transform, params, tile_size);
    } else if (params.features_ % 2 == 0) {
      return LaunchVectorTransform<T, 2, CType>::launch(
          device, input, in_offset, transform, params, tile_size);
    } else {
      return LaunchVectorTransform<T, 1, CType>::launch(
          device, input, in_offset, transform, params, tile_size);
    }
  }
};
template <typename T>
struct LaunchIm2colTransform<T, ConvType::FilterBackprop> {
  using Index = int;
  static constexpr auto CType = ConvType::FilterBackprop;
  static cl::sycl::event launch(Eigen::SyclDevice const& device,
                                T const* const input, Index const in_offset,
                                T* const transform,
                                const SYCLConv2DParams& params,
                                Index const tile_size) {
    return LaunchVectorTransform<T, 1, CType>::launch(
        device, input, in_offset, transform, params, tile_size);
  }
};
template <typename T, ConvType CType>
struct RunIm2ColAllocated;
template <typename T>
struct RunIm2ColAllocated<T, ConvType::Forward> {
  using Index = int;
  static constexpr auto CType = ConvType::Forward;

  static void run(Eigen::SyclDevice const& device, T* const output,
                  Index const out_offset, T const* const input,
                  Index const in_offset, T const* const filter,
                  T* const transform, const SYCLConv2DParams& params,
                  TileInfo tile_info) {
    const size_t alloc_size_per_image =
        tile_info.number * tile_info.size * sizeof(T);
    const Index n_tiles = params.batch_ * tile_info.number;
    const size_t actual_transform_size = alloc_size_per_image * params.batch_;

    device.memset(transform, 0, actual_transform_size);
    LaunchIm2colTransform<T, CType>::launch(device, input, in_offset, transform,
                                            params, tile_info.size);
    sycl_conv::launch_matmul<false, false>(
        device, transform, filter, output + out_offset, static_cast<T>(0),
        n_tiles, tile_info.size, params.features_);
  }
};
template <typename T>
struct RunIm2ColAllocated<T, ConvType::InputBackprop> {
  using Index = int;
  static constexpr auto CType = ConvType::InputBackprop;

  static void run(Eigen::SyclDevice const& device, T* const output,
                  Index const out_offset, T const* const input,
                  Index const in_offset, T const* const filter_transform,
                  T* const transform, const SYCLConv2DParams& params,
                  TileInfo tile_info) {
    const size_t alloc_size_per_image =
        tile_info.number * tile_info.size * sizeof(T);
    const Index n_tiles = params.batch_ * tile_info.number;
    const size_t actual_transform_size = alloc_size_per_image * params.batch_;

    device.memset(transform, 0, actual_transform_size);
    LaunchIm2colTransform<T, CType>::launch(device, input, in_offset, transform,
                                            params, tile_info.size);
    sycl_conv::launch_matmul<false, false>(
        device, transform, filter_transform, output + out_offset,
        static_cast<T>(0), n_tiles, tile_info.size, params.channels_);
  }
};
template <typename T>
struct RunIm2ColAllocated<T, ConvType::FilterBackprop> {
  using Index = int;
  static constexpr auto CType = ConvType::FilterBackprop;

  static void run(Eigen::SyclDevice const& device, T* const output,
                  Index const out_offset, T const* const input,
                  Index const in_offset, T const* const filter,
                  T* const transform, SYCLConv2DParams const& params,
                  TileInfo tile_info) {
    // Here the tensors are assumed to be the following:
    //  - 'input' is the original input to the initial convolution.
    //  - 'output' is the filter backprop tensor.
    //  - 'filter' is the output of the initial convolution, used here as the
    // filter over the input.
    //
    //  The params expected is the kernel params, which has:
    //   - window dims given by the size of the 'filter' tensor.
    //   - out dims given by the size of the 'output' tensor.
    const Index n_tiles = tile_info.number;
    const Index tile_size = params.batch_ * tile_info.size;
    const size_t alloc_size_per_image =
        tile_info.size * tile_info.number * sizeof(T);
    const size_t actual_transform_size = alloc_size_per_image * params.batch_;

    device.memset(transform, 0, actual_transform_size);
    LaunchIm2colTransform<T, CType>::launch(device, input, in_offset, transform,
                                            params, tile_size);
    if (in_offset == 0) {
      sycl_conv::launch_matmul<false, false>(
          device, transform, filter + out_offset, output, static_cast<T>(0),
          n_tiles, tile_size, params.features_);
    } else {
      sycl_conv::launch_matmul<false, false>(
          device, transform, filter + out_offset, output, static_cast<T>(1),
          n_tiles, tile_size, params.features_);
      // For Eigen, this matmul with non-zero alpha will trigger an
      // allocation. We need to ensure that we synchronize here to prevent
      // allocation failures with large buffers.
      device.synchronize();
    }
  }
};
template <ConvType CType>
inline SYCLConv2DParams get_params(SYCLConv2DParams params) {
  return params;
}
template <>
inline SYCLConv2DParams get_params<ConvType::FilterBackprop>(
    SYCLConv2DParams params) {
  std::swap(params.out_rows_, params.window_rows_);
  std::swap(params.out_cols_, params.window_cols_);
  std::swap(params.stride_rows_, params.dilation_rows_);
  std::swap(params.stride_cols_, params.dilation_cols_);
  return params;
}
}
template <typename T, ConvType CType>
struct LaunchIm2Col {
  using Index = int;

  static bool launch(Eigen::SyclDevice const& device, T* const output,
                     T const* const input, T const* const filter,
                     const SYCLConv2DParams& params) {
    cl::sycl::queue& sycl_queue = device.sycl_queue();
    cl::sycl::device const& sycl_device = sycl_queue.get_device();

    const im2col::TileInfo tile_info = im2col::get_tile_info<CType>(params);

    T const* const filter_transform =
        im2col::FilterTransformAllocator<T, CType>::get_transform(
            device, filter, params);

    const size_t alloc_size_per_image =
        tile_info.number * tile_info.size * sizeof(T);
    const im2col::AllocInfo alloc_info = im2col::get_alloc_info<CType>(
        sycl_queue, sycl_device, params.batch_, alloc_size_per_image);
    size_t images_per_alloc = alloc_info.images_per_alloc;
    if (alloc_info.alloc_warning) {
      return false;
    }

    SYCLConv2DParams kernel_params = im2col::get_params<CType>(params);
    // When the number of input transforms required is greater than one, it is
    // because each transform buffer is right on the limit of what can be
    // allocated on the device. This means that we need to reuse the same
    // buffer for each iteration of the algorithm, as enqueuing more than one
    // of these huge buffers can cause CL_MEM_OBJECT_ALLOCATION_FAILURE errors.
    // Reusing the same buffer will add a little overhead, as the iterations
    // cannot be run in parallel however it is required to avoid the allocation
    // failures.
    // TODO(jwlawson): Remove when sycl queue can handle this for us.
    const size_t transform_size = alloc_size_per_image * images_per_alloc;
    T* const transform = static_cast<T*>(device.allocate(transform_size));

    for (int i = 0; i < alloc_info.n_input_transforms; ++i) {
      im2col::Offsets offset =
          im2col::calculate_offsets<CType>(i, images_per_alloc, params);
      assert(i == 0 || in_offset > 0);
      assert(i == 0 || out_offset > 0);
      if (i == alloc_info.n_input_transforms - 1) {
        images_per_alloc = alloc_info.last_batch_size;
      }
      kernel_params.batch_ = images_per_alloc;
      im2col::RunIm2ColAllocated<T, CType>::run(
          device, output, offset.out, input, offset.in, filter_transform,
          transform, kernel_params, tile_info);
    }
    device.deallocate(transform);
    im2col::FilterTransformAllocator<T, CType>::deallocate(device,
                                                           filter_transform);
    // At the moment we have to explicitly wait here to ensure that the device
    // queue is cleared before enqueuing the kernels which use huge buffers so
    // that we do not hit any memory allocation failures.
    // TODO(jwlawson): Remove wait when SYCL queue handles allocation waiting.
    device.synchronize();
    return true;
  }
};
template <typename T, typename backend_type, ConvType CType>
struct Launcher<T, backend_type, algorithm::im2col, CType> final
    : public LaunchIm2Col<T, CType> {};
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_IM2COL_SYCL_H_
