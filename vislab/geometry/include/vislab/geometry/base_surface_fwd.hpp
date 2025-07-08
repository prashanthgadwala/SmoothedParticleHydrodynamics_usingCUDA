#pragma once

#include <stdint.h>

namespace vislab
{
    /**
     * @brief Base class for surface geometry with a certain dimensionality.
     * @tparam TDimensions Number of dimensions.
     */
    template <int64_t TDimensions>
    class BaseSurface;

    /**
     * @brief Interface for one-dimensional surface geometry.
     */
    using ISurface1 = BaseSurface<1>;

    /**
     * @brief Interface for two-dimensional surface geometry.
     */
    using ISurface2 = BaseSurface<2>;

    /**
     * @brief Interface for three-dimensional surface geometry.
     */
    using ISurface3 = BaseSurface<3>;

    /**
     * @brief Interface for four-dimensional surface geometry.
     */
    using ISurface4 = BaseSurface<4>;
}
