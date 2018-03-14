#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_SYCL_SELECTORS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_SYCL_SELECTORS_H_

#include "tensorflow/core/kernels/conv_ops_sycl_common.h"

namespace tensorflow {
class algorithm_selector {
 public:
  virtual algorithm get_selection(SYCLConv2DParams const&) = 0;
};
template <algorithm Selection>
class constant_selector final : public algorithm_selector {
 public:
  algorithm get_selection(SYCLConv2DParams const&) override {
    return Selection;
  }
};
using direct_selector = constant_selector<algorithm::direct>;
using im2col_selector = constant_selector<algorithm::im2col>;
class winograd_selector final : public algorithm_selector {
 public:
  algorithm get_selection(SYCLConv2DParams const& params) override {
    if (params.stride_rows_ != 1 || params.stride_cols_ != 1) {
      return algorithm::not_supported;
    }
    if (params.window_rows_ == 1 && params.window_cols_ == 3) {
      return algorithm::winograd_1x3;
    }
    if (params.window_rows_ == 3 && params.window_cols_ == 1) {
      return algorithm::winograd_3x1;
    }
    if (params.window_rows_ == 3 && params.window_cols_ == 3) {
      return algorithm::winograd_3x3;
    }
    return algorithm::not_supported;
  }
};
class direct_tiled_selector final : public algorithm_selector {
 public:
  algorithm get_selection(SYCLConv2DParams const& params) override {
    if (params.window_rows_ != params.window_cols_ ||
        params.stride_rows_ != params.stride_cols_) {
      return algorithm::not_supported;
    }
    if (params.window_rows_ == 1 && params.stride_rows_ == 2) {
      return algorithm::direct_tiled;
    }
    if (params.window_rows_ == 1 && params.stride_rows_ == 1) {
      return algorithm::direct_tiled;
    }
    if (params.window_rows_ == 3 && params.stride_rows_ == 2) {
      return algorithm::direct_tiled;
    }
    if (params.window_rows_ == 3 && params.stride_rows_ == 1) {
      return algorithm::direct_tiled;
    }
    if (params.window_rows_ == 5 && params.stride_rows_ == 1) {
      return algorithm::direct_tiled;
    }
    return algorithm::not_supported;
  }
};
class matmul_selector final : public algorithm_selector {
 public:
  algorithm get_selection(SYCLConv2DParams const& params) override {
    if (params.window_rows_ == 1 && params.window_cols_ == 1 &&
        params.stride_rows_ == 1 && params.stride_cols_ == 1) {
      return algorithm::matmul;
    }
    return algorithm::not_supported;
  }
};
class arm_selector final : public algorithm_selector {
 public:
  algorithm get_selection(SYCLConv2DParams const& params) override {
    if (params.window_rows_ == params.window_cols_ && params.stride_rows_ == params.stride_cols_) {
      if(params.window_rows_ == 1 && params.stride_rows_ == 2) {
        return algorithm::direct_tiled;
      }
      if(params.window_rows_ == 1 && params.stride_rows_ == 1) {
        return algorithm::direct_tiled;
      }
      if(params.window_rows_ == 3 && params.stride_rows_ == 1) {
        if(params.channels_ < 10) {
          return algorithm::direct;
        } else if( params.in_rows_ > 100) {
          return algorithm::winograd_3x3;
        } else if (params.in_rows_ > 50) {
          return algorithm::im2col;
        } else {
          return algorithm::direct_tiled;
        }
      }
    }
    return algorithm::direct;
  }
};

template <typename Initial, typename Fallback>
class fallback_selector final : public algorithm_selector {
  static_assert(std::is_base_of<algorithm_selector, Initial>::value,
                "fallback_selector expects the 'Initial' template parameter to "
                "be derived from 'algorithm_selector'.");
  static_assert(std::is_base_of<algorithm_selector, Fallback>::value,
                "fallback_selector expects the 'Fallback' template parameter "
                "to be derived from 'algorithm_selector'.");
  // TODO(jwlawson): Provide forwarding constructors
 public:
  algorithm get_selection(SYCLConv2DParams const& params) override {
    algorithm initial_selection = initial_selector_.get_selection(params);
    if (initial_selection != algorithm::not_supported) {
      return initial_selection;
    }
    return fallback_selector_.get_selection(params);
  }

 private:
  Initial initial_selector_;
  Fallback fallback_selector_;
};

template <typename Selector>
using matmul_then = fallback_selector<matmul_selector, Selector>;
template <typename Selector>
using winograd_then = fallback_selector<winograd_selector, Selector>;
template <typename Selector>
using direct_tiled_then = fallback_selector<direct_tiled_selector, Selector>;
using default_selector = matmul_then<winograd_then<im2col_selector>>;
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_SELECTORS_H_
