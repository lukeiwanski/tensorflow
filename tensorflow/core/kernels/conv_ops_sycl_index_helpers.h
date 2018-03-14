#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_SYCL_INDEX_HELPERS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_SYCL_INDEX_HELPERS_H_

#include "tensorflow/core/kernels/conv_ops_sycl_fast_div.h"
#include "tensorflow/core/kernels/conv_ops_sycl_kernel_macros.h"

namespace tensorflow {
namespace helpers {
/**
 * Get the index at which the window starts in the input tensor for the given
 * output index.
 */
template <typename Index>
inline SNN_ALWAYS_INLINE Index out_window_start(Index const in,
                                                Index const stride,
                                                Index const pad) {
  return in * stride - pad;
}
/**
 * Get the index in the window at which the output index starts.
 */
template <typename Index>
inline SNN_ALWAYS_INLINE Index out_filter_start(Index const /*in*/,
                                                Index const /*stride*/,
                                                Index const /*pad*/) {
  return 0;
}
/**
 * Get the index at which the window starts in the output tensor for the given
 * input index.
 */
/* The padding here is expected to be the output padding.
 * (pad_out = window - 1 - pad_in) */
template <typename Index>
inline SNN_ALWAYS_INLINE Index in_window_start(Index const in,
                                               Index const stride,
                                               Index const pad) {
  return RoundRatioUp(in - pad, stride);
}
/**
 * Get the index in the window at which the input index starts.
 */
template <typename Index>
inline SNN_ALWAYS_INLINE Index in_filter_start(Index const in,
                                               Index const stride,
                                               Index const pad) {
  const Index r = in - pad;
  const Index start = (r < 0 ? -r : stride - r) % stride;
  return (start < 0 ? -start : start);
}
// The following provide an alternative approach to computing the start values
// for the input window given an output index. This requires fewer branches than
// the approach above, but will always round the window start up to zero if it
// is negative.
template <typename Index>
inline SNN_ALWAYS_INLINE Index in_window_start_above_zero(Index const in,
                                                          Index const stride,
                                                          Index const pad) {
  return RoundRatioUpAboveZero(in - pad, stride);
}
template <typename Index>
inline SNN_ALWAYS_INLINE Index in_filter_start_above_zero(Index const in,
                                                          Index const stride,
                                                          Index const pad) {
  const Index window_start = in_window_start_above_zero(in, stride, pad);
  const Index padded = in - pad;
  return window_start * stride - padded;
}

}  // namespace helpers
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_INDEX_HELPERS_H_
