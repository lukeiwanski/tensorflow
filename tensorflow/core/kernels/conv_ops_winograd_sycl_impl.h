#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_IMPL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_IMPL_H_

namespace tensorflow {
namespace winograd {
/**
 * For any Winograd tiling you wish to implement, ensure that the following
 * three specialisations are completed:
template <typename T>
struct TransformedFilterTile<T, 2, 2, 3, 3> final
    : public BaseTransformedFilterTile<T, 2, 2, 3, 3> {
  using BaseTransformedFilterTile<T, 2, 2, 3, 3>::data;

  template <ConvType _FT>
  inline SNN_ALWAYS_INLINE TransformedFilterTile(
      FilterTile<T, 2, 2, 3, 3, _FT> const& filter)
      : BaseTransformedFilterTile<T, 2, 2, 3, 3>{} {
    // ...
  }
};

template <typename T>
struct TransformedInputTile<T, 2, 2, 3, 3> final
    : public BaseTransformedInputTile<T, 2, 2, 3, 3> {
  using BaseTransformedInputTile<T, 2, 2, 3, 3>::data;

  inline SNN_ALWAYS_INLINE TransformedInputTile(
      InputTile<T, 2, 2, 3, 3> const& inp)
      : BaseTransformedInputTile<T, 2, 2, 3, 3>{} {
    // ...
  }
};

template <typename T>
struct OutputTile<T, 2, 2, 3, 3> final : public BaseOutputTile<T, 2, 2, 3, 3> {
  using BaseOutputTile<T, 2, 2, 3, 3>::data;
  inline SNN_ALWAYS_INLINE OutputTile(
      IntermediateTile<T, 2, 2, 3, 3> const& tile)
      : BaseOutputTile<T, 2, 2, 3, 3>{} {
    // ...
  }
};
*/
template <typename T>
struct TransformedFilterTile<T, 2, 2, 3, 3> final
    : public BaseTransformedFilterTile<T, 2, 2, 3, 3> {
  using BaseTransformedFilterTile<T, 2, 2, 3, 3>::data;
  /**
   * Apply the Winograd transform to the filter tile.
   */
  template <ConvType _FT>
  inline SNN_ALWAYS_INLINE TransformedFilterTile(
      FilterTile<T, 2, 2, 3, 3, _FT> const& filter)
      : BaseTransformedFilterTile<T, 2, 2, 3, 3>{} {
    data[0][0] = filter.data[0][0];
    data[0][1] =
        (filter.data[0][0] + filter.data[0][1] + filter.data[0][2]) / 2;
    data[0][2] =
        (filter.data[0][0] - filter.data[0][1] + filter.data[0][2]) / 2;
    data[0][3] = filter.data[0][2];

    data[1][0] =
        (filter.data[0][0] + filter.data[1][0] + filter.data[2][0]) / 2;
    data[1][1] = (filter.data[0][0] + filter.data[0][1] + filter.data[0][2] +
                  filter.data[1][0] + filter.data[1][1] + filter.data[1][2] +
                  filter.data[2][0] + filter.data[2][1] + filter.data[2][2]) /
                 4;
    data[1][2] = (filter.data[0][0] - filter.data[0][1] + filter.data[0][2] +
                  filter.data[1][0] - filter.data[1][1] + filter.data[1][2] +
                  filter.data[2][0] - filter.data[2][1] + filter.data[2][2]) /
                 4;
    data[1][3] =
        (filter.data[0][2] + filter.data[1][2] + filter.data[2][2]) / 2;

    data[2][0] =
        (filter.data[0][0] - filter.data[1][0] + filter.data[2][0]) / 2;
    data[2][1] = (filter.data[0][0] + filter.data[0][1] + filter.data[0][2] -
                  filter.data[1][0] - filter.data[1][1] - filter.data[1][2] +
                  filter.data[2][0] + filter.data[2][1] + filter.data[2][2]) /
                 4;
    data[2][2] = (filter.data[0][0] - filter.data[0][1] + filter.data[0][2] -
                  filter.data[1][0] + filter.data[1][1] - filter.data[1][2] +
                  filter.data[2][0] - filter.data[2][1] + filter.data[2][2]) /
                 4;
    data[2][3] =
        (filter.data[0][2] - filter.data[1][2] + filter.data[2][2]) / 2;

    data[3][0] = filter.data[2][0];
    data[3][1] =
        (filter.data[2][0] + filter.data[2][1] + filter.data[2][2]) / 2;
    data[3][2] =
        (filter.data[2][0] - filter.data[2][1] + filter.data[2][2]) / 2;
    data[3][3] = filter.data[2][2];
  }
};
template <typename T>
struct TransformedInputTile<T, 2, 2, 3, 3> final
    : public BaseTransformedInputTile<T, 2, 2, 3, 3> {
  using BaseTransformedInputTile<T, 2, 2, 3, 3>::data;
  /**
   * Apply the Winograd transform to the filter tile.
   */
  inline SNN_ALWAYS_INLINE TransformedInputTile(
      InputTile<T, 2, 2, 3, 3> const& inp)
      : BaseTransformedInputTile<T, 2, 2, 3, 3>{} {
    data[0][0] =
        inp.data[0][0] + inp.data[2][2] - inp.data[0][2] - inp.data[2][0];
    data[0][1] =
        inp.data[0][1] + inp.data[0][2] - inp.data[2][1] - inp.data[2][2];
    data[0][2] =
        inp.data[0][2] + inp.data[2][1] - inp.data[0][1] - inp.data[2][2];
    data[0][3] =
        inp.data[0][3] + inp.data[2][1] - inp.data[0][1] - inp.data[2][3];

    data[1][0] =
        inp.data[1][0] + inp.data[2][0] - inp.data[1][2] - inp.data[2][2];
    data[1][1] =
        inp.data[1][1] + inp.data[1][2] + inp.data[2][1] + inp.data[2][2];
    data[1][2] =
        inp.data[1][2] + inp.data[2][2] - inp.data[1][1] - inp.data[2][1];
    data[1][3] =
        inp.data[1][3] + inp.data[2][3] - inp.data[1][1] - inp.data[2][1];

    data[2][0] =
        inp.data[1][2] + inp.data[2][0] - inp.data[1][0] - inp.data[2][2];
    data[2][1] =
        inp.data[2][1] + inp.data[2][2] - inp.data[1][1] - inp.data[1][2];
    data[2][2] =
        inp.data[1][1] + inp.data[2][2] - inp.data[1][2] - inp.data[2][1];
    data[2][3] =
        inp.data[1][1] + inp.data[2][3] - inp.data[1][3] - inp.data[2][1];

    data[3][0] =
        inp.data[1][2] + inp.data[3][0] - inp.data[1][0] - inp.data[3][2];
    data[3][1] =
        inp.data[3][1] + inp.data[3][2] - inp.data[1][1] - inp.data[1][2];
    data[3][2] =
        inp.data[1][1] + inp.data[3][2] - inp.data[1][2] - inp.data[3][1];
    data[3][3] =
        inp.data[1][1] + inp.data[3][3] - inp.data[1][3] - inp.data[3][1];
  }
};
template <typename T>
struct OutputTile<T, 2, 2, 3, 3> final : public BaseOutputTile<T, 2, 2, 3, 3> {
  using BaseOutputTile<T, 2, 2, 3, 3>::data;
  /**
   * Apply the Winograd transform to the intermediate tile to give the final
   * output tile.
   */
  inline SNN_ALWAYS_INLINE OutputTile(
      IntermediateTile<T, 2, 2, 3, 3> const& tile)
      : BaseOutputTile<T, 2, 2, 3, 3>{} {
    data[0][0] = tile.data[0][0] + tile.data[0][1] + tile.data[0][2] +
                 tile.data[1][0] + tile.data[1][1] + tile.data[1][2] +
                 tile.data[2][0] + tile.data[2][1] + tile.data[2][2];
    data[0][1] = tile.data[0][1] - tile.data[0][2] + tile.data[0][3] +
                 tile.data[1][1] - tile.data[1][2] + tile.data[1][3] +
                 tile.data[2][1] - tile.data[2][2] + tile.data[2][3];
    data[1][0] = tile.data[1][0] + tile.data[1][1] + tile.data[1][2] -
                 tile.data[2][0] - tile.data[2][1] - tile.data[2][2] +
                 tile.data[3][0] + tile.data[3][1] + tile.data[3][2];
    data[1][1] = tile.data[1][1] - tile.data[1][2] + tile.data[1][3] -
                 tile.data[2][1] + tile.data[2][2] - tile.data[2][3] +
                 tile.data[3][1] - tile.data[3][2] + tile.data[3][3];
  }
};
template <typename T>
struct TransformedFilterTile<T, 2, 1, 3, 1> final
    : public BaseTransformedFilterTile<T, 2, 1, 3, 1> {
  using BaseTransformedFilterTile<T, 2, 1, 3, 1>::data;

  template <ConvType _FT>
  inline SNN_ALWAYS_INLINE TransformedFilterTile(
      FilterTile<T, 2, 1, 3, 1, _FT> const& filter)
      : BaseTransformedFilterTile<T, 2, 1, 3, 1>{} {
    data[0][0] = filter.data[0][0];
    data[1][0] =
        (filter.data[0][0] + filter.data[1][0] + filter.data[2][0]) / 2;
    data[2][0] =
        (filter.data[0][0] - filter.data[1][0] + filter.data[2][0]) / 2;
    data[3][0] = filter.data[2][0];
  }
};

template <typename T>
struct TransformedInputTile<T, 2, 1, 3, 1> final
    : public BaseTransformedInputTile<T, 2, 1, 3, 1> {
  using BaseTransformedInputTile<T, 2, 1, 3, 1>::data;

  inline SNN_ALWAYS_INLINE TransformedInputTile(
      InputTile<T, 2, 1, 3, 1> const& inp)
      : BaseTransformedInputTile<T, 2, 1, 3, 1>{} {
    data[0][0] = inp.data[0][0] - inp.data[2][0];
    data[1][0] = inp.data[1][0] + inp.data[2][0];
    data[2][0] = inp.data[2][0] - inp.data[1][0];
    data[3][0] = inp.data[3][0] - inp.data[1][0];
  }
};

template <typename T>
struct OutputTile<T, 2, 1, 3, 1> final : public BaseOutputTile<T, 2, 1, 3, 1> {
  using BaseOutputTile<T, 2, 1, 3, 1>::data;
  inline SNN_ALWAYS_INLINE OutputTile(
      IntermediateTile<T, 2, 1, 3, 1> const& tile)
      : BaseOutputTile<T, 2, 1, 3, 1>{} {
    data[0][0] = tile.data[0][0] + tile.data[1][0] + tile.data[2][0];
    data[1][0] = tile.data[1][0] - tile.data[2][0] + tile.data[3][0];
  }
};
template <typename T>
struct TransformedFilterTile<T, 1, 2, 1, 3> final
    : public BaseTransformedFilterTile<T, 1, 2, 1, 3> {
  using BaseTransformedFilterTile<T, 1, 2, 1, 3>::data;

  template <ConvType _FT>
  inline SNN_ALWAYS_INLINE TransformedFilterTile(
      FilterTile<T, 1, 2, 1, 3, _FT> const& filter)
      : BaseTransformedFilterTile<T, 1, 2, 1, 3>{} {
    data[0][0] = filter.data[0][0];
    data[0][1] =
        (filter.data[0][0] + filter.data[0][1] + filter.data[0][2]) / 2;
    data[0][2] =
        (filter.data[0][0] - filter.data[0][1] + filter.data[0][2]) / 2;
    data[0][3] = filter.data[0][2];
  }
};

template <typename T>
struct TransformedInputTile<T, 1, 2, 1, 3> final
    : public BaseTransformedInputTile<T, 1, 2, 1, 3> {
  using BaseTransformedInputTile<T, 1, 2, 1, 3>::data;

  inline SNN_ALWAYS_INLINE TransformedInputTile(
      InputTile<T, 1, 2, 1, 3> const& inp)
      : BaseTransformedInputTile<T, 1, 2, 1, 3>{} {
    data[0][0] = inp.data[0][0] - inp.data[0][2];
    data[0][1] = inp.data[0][1] + inp.data[0][2];
    data[0][2] = inp.data[0][2] - inp.data[0][1];
    data[0][3] = inp.data[0][3] - inp.data[0][1];
  }
};

template <typename T>
struct OutputTile<T, 1, 2, 1, 3> final : public BaseOutputTile<T, 1, 2, 1, 3> {
  using BaseOutputTile<T, 1, 2, 1, 3>::data;
  inline SNN_ALWAYS_INLINE OutputTile(
      IntermediateTile<T, 1, 2, 1, 3> const& tile)
      : BaseOutputTile<T, 1, 2, 1, 3>{} {
    data[0][0] = tile.data[0][0] + tile.data[0][1] + tile.data[0][2];
    data[0][1] = tile.data[0][1] - tile.data[0][2] + tile.data[0][3];
  }
};
template <typename T>
struct TransformedFilterTile<T, 3, 1, 2, 1> final
    : public BaseTransformedFilterTile<T, 3, 1, 2, 1> {
  using BaseTransformedFilterTile<T, 3, 1, 2, 1>::data;

  template <ConvType _FT>
  inline SNN_ALWAYS_INLINE TransformedFilterTile(
      FilterTile<T, 3, 1, 2, 1, _FT> const& filter)
      : BaseTransformedFilterTile<T, 3, 1, 2, 1>{} {
    data[0][0] = filter.data[0][0];
    data[1][0] = (filter.data[0][0] + filter.data[1][0]) / 2;
    data[2][0] = (filter.data[0][0] - filter.data[1][0]) / 2;
    data[3][0] = filter.data[1][0];
  }
};

template <typename T>
struct TransformedInputTile<T, 3, 1, 2, 1> final
    : public BaseTransformedInputTile<T, 3, 1, 2, 1> {
  using BaseTransformedInputTile<T, 3, 1, 2, 1>::data;

  inline SNN_ALWAYS_INLINE TransformedInputTile(
      InputTile<T, 3, 1, 2, 1> const& inp)
      : BaseTransformedInputTile<T, 3, 1, 2, 1>{} {
    data[0][0] = inp.data[0][0] - inp.data[2][0];
    data[1][0] = inp.data[1][0] + inp.data[2][0];
    data[2][0] = -inp.data[1][0] + inp.data[2][0];
    data[3][0] = -inp.data[1][0] + inp.data[3][0];
  }
};

template <typename T>
struct OutputTile<T, 3, 1, 2, 1> final : public BaseOutputTile<T, 3, 1, 2, 1> {
  using BaseOutputTile<T, 3, 1, 2, 1>::data;
  inline SNN_ALWAYS_INLINE OutputTile(
      IntermediateTile<T, 3, 1, 2, 1> const& tile)
      : BaseOutputTile<T, 3, 1, 2, 1>{} {
    data[0][0] = tile.data[0][0] + tile.data[1][0] + tile.data[2][0];
    data[1][0] = tile.data[1][0] - tile.data[2][0];
    data[2][0] = tile.data[1][0] + tile.data[2][0] + tile.data[3][0];
  }
};
template <typename T>
struct TransformedFilterTile<T, 1, 3, 1, 2> final
    : public BaseTransformedFilterTile<T, 1, 3, 1, 2> {
  using BaseTransformedFilterTile<T, 1, 3, 1, 2>::data;

  template <ConvType _FT>
  inline SNN_ALWAYS_INLINE TransformedFilterTile(
      FilterTile<T, 1, 3, 1, 2, _FT> const& filter)
      : BaseTransformedFilterTile<T, 1, 3, 1, 2>{} {
    data[0][0] = filter.data[0][0];
    data[0][1] = (filter.data[0][0] + filter.data[0][1]) / 2;
    data[0][2] = (filter.data[0][0] - filter.data[0][1]) / 2;
    data[0][3] = filter.data[0][1];
  }
};

template <typename T>
struct TransformedInputTile<T, 1, 3, 1, 2> final
    : public BaseTransformedInputTile<T, 1, 3, 1, 2> {
  using BaseTransformedInputTile<T, 1, 3, 1, 2>::data;

  inline SNN_ALWAYS_INLINE TransformedInputTile(
      InputTile<T, 1, 3, 1, 2> const& inp)
      : BaseTransformedInputTile<T, 1, 3, 1, 2>{} {
    data[0][0] = inp.data[0][0] - inp.data[0][2];
    data[0][1] = inp.data[0][1] + inp.data[0][2];
    data[0][2] = -inp.data[0][1] + inp.data[0][2];
    data[0][3] = -inp.data[0][1] + inp.data[0][3];
  }
};

template <typename T>
struct OutputTile<T, 1, 3, 1, 2> final : public BaseOutputTile<T, 1, 3, 1, 2> {
  using BaseOutputTile<T, 1, 3, 1, 2>::data;
  inline SNN_ALWAYS_INLINE OutputTile(
      IntermediateTile<T, 1, 3, 1, 2> const& tile)
      : BaseOutputTile<T, 1, 3, 1, 2>{} {
    data[0][0] = tile.data[0][0] + tile.data[0][1] + tile.data[0][2];
    data[0][1] = tile.data[0][1] - tile.data[0][2];
    data[0][2] = tile.data[0][1] + tile.data[0][2] + tile.data[0][3];
  }
};

template <typename T>
struct TransformedFilterTile<T, 3, 3, 2, 2> final
    : public BaseTransformedFilterTile<T, 3, 3, 2, 2> {
  using BaseTransformedFilterTile<T, 3, 3, 2, 2>::data;

  template <ConvType _FT>
  inline SNN_ALWAYS_INLINE TransformedFilterTile(
      FilterTile<T, 3, 3, 2, 2, _FT> const& filter)
      : BaseTransformedFilterTile<T, 3, 3, 2, 2>{} {
    data[0][0] = filter.data[0][0];
    data[0][1] = (filter.data[0][0] + filter.data[0][1]) / 2;
    data[0][2] = (filter.data[0][0] - filter.data[0][1]) / 2;
    data[0][3] = filter.data[0][1];

    data[1][0] = (filter.data[0][0] + filter.data[1][0]) / 2;
    data[1][1] = (filter.data[0][0] + filter.data[0][1] + filter.data[1][0] +
                  filter.data[1][1]) /
                 4;
    data[1][2] = (filter.data[0][0] - filter.data[0][1] + filter.data[1][0] -
                  filter.data[1][1]) /
                 4;
    data[1][3] = (filter.data[0][1] + filter.data[1][1]) / 2;

    data[2][0] = (filter.data[0][0] - filter.data[1][0]) / 2;
    data[2][1] = (filter.data[0][0] + filter.data[0][1] - filter.data[1][0] -
                  filter.data[1][1]) /
                 4;
    data[2][2] = (filter.data[0][0] - filter.data[0][1] - filter.data[1][0] +
                  filter.data[1][1]) /
                 4;
    data[2][3] = (filter.data[0][1] - filter.data[1][1]) / 2;

    data[3][0] = filter.data[1][0];
    data[3][1] = (filter.data[1][0] + filter.data[1][1]) / 2;
    data[3][2] = (filter.data[1][0] - filter.data[1][1]) / 2;
    data[3][3] = filter.data[1][1];
  }
};
template <typename T>
struct TransformedInputTile<T, 3, 3, 2, 2> final
    : public BaseTransformedInputTile<T, 3, 3, 2, 2> {
  using BaseTransformedInputTile<T, 3, 3, 2, 2>::data;

  inline SNN_ALWAYS_INLINE TransformedInputTile(
      InputTile<T, 3, 3, 2, 2> const& inp)
      : BaseTransformedInputTile<T, 3, 3, 2, 2>{} {
    data[0][0] =
        inp.data[0][0] - inp.data[0][2] - inp.data[2][0] + inp.data[2][2];
    data[0][1] =
        inp.data[0][1] + inp.data[0][2] - inp.data[2][1] - inp.data[2][2];
    data[0][2] =
        -inp.data[0][1] + inp.data[0][2] + inp.data[2][1] - inp.data[2][2];
    data[0][3] =
        -inp.data[0][1] + inp.data[0][3] + inp.data[2][1] - inp.data[2][3];

    data[1][0] =
        inp.data[1][0] - inp.data[1][2] + inp.data[2][0] - inp.data[2][2];
    data[1][1] =
        inp.data[1][1] + inp.data[1][2] + inp.data[2][1] + inp.data[2][2];
    data[1][2] =
        -inp.data[1][1] + inp.data[1][2] - inp.data[2][1] + inp.data[2][2];
    data[1][3] =
        -inp.data[1][1] + inp.data[1][3] - inp.data[2][1] + inp.data[2][3];

    data[2][0] =
        -inp.data[1][0] + inp.data[1][2] + inp.data[2][0] - inp.data[2][2];
    data[2][1] =
        -inp.data[1][1] - inp.data[1][2] + inp.data[2][1] + inp.data[2][2];
    data[2][2] =
        inp.data[1][1] - inp.data[1][2] - inp.data[2][1] + inp.data[2][2];
    data[2][3] =
        inp.data[1][1] - inp.data[1][3] - inp.data[2][1] + inp.data[2][3];

    data[3][0] =
        -inp.data[1][0] + inp.data[1][2] + inp.data[3][0] - inp.data[3][2];
    data[3][1] =
        -inp.data[1][1] - inp.data[1][2] + inp.data[3][1] + inp.data[3][2];
    data[3][2] =
        inp.data[1][1] - inp.data[1][2] - inp.data[3][1] + inp.data[3][2];
    data[3][3] =
        inp.data[1][1] - inp.data[1][3] - inp.data[3][1] + inp.data[3][3];
  }
};

template <typename T>
struct OutputTile<T, 3, 3, 2, 2> final : public BaseOutputTile<T, 3, 3, 2, 2> {
  using BaseOutputTile<T, 3, 3, 2, 2>::data;
  inline SNN_ALWAYS_INLINE OutputTile(
      IntermediateTile<T, 3, 3, 2, 2> const& tile)
      : BaseOutputTile<T, 3, 3, 2, 2>{} {
    data[0][0] = tile.data[0][0] + tile.data[0][1] + tile.data[0][2] +
                 tile.data[1][0] + tile.data[1][1] + tile.data[1][2] +
                 tile.data[2][0] + tile.data[2][1] + tile.data[2][2];
    data[0][1] = tile.data[0][1] - tile.data[0][2] + tile.data[1][1] -
                 tile.data[1][2] + tile.data[2][1] - tile.data[2][2];
    data[0][2] = tile.data[0][1] + tile.data[0][2] + tile.data[0][3] +
                 tile.data[1][1] + tile.data[1][2] + tile.data[1][3] +
                 tile.data[2][1] + tile.data[2][2] + tile.data[2][3];

    data[1][0] = tile.data[1][0] + tile.data[1][1] + tile.data[1][2] -
                 tile.data[2][0] - tile.data[2][1] - tile.data[2][2];
    data[1][1] =
        tile.data[1][1] - tile.data[1][2] - tile.data[2][1] + tile.data[2][2];
    data[1][2] = tile.data[1][1] + tile.data[1][2] + tile.data[1][3] -
                 tile.data[2][1] - tile.data[2][2] - tile.data[2][3];

    data[2][0] = tile.data[1][0] + tile.data[1][1] + tile.data[1][2] +
                 tile.data[2][0] + tile.data[2][1] + tile.data[2][2] +
                 tile.data[3][0] + tile.data[3][1] + tile.data[3][2];
    data[2][1] = tile.data[1][1] - tile.data[1][2] + tile.data[2][1] -
                 tile.data[2][2] + tile.data[3][1] - tile.data[3][2];
    data[2][2] = tile.data[1][1] + tile.data[1][2] + tile.data[1][3] +
                 tile.data[2][1] + tile.data[2][2] + tile.data[2][3] +
                 tile.data[3][1] + tile.data[3][2] + tile.data[3][3];
  }
};
}  // namespace winograd
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_WINOGRAD_SYCL_IMPL_H_
