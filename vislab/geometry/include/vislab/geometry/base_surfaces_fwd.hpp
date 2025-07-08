#pragma once

#include <stdint.h>

namespace vislab
{
    /**
     * @brief Base class for set of surface geometries with a certain dimensionality.
     * @tparam TDimensions Number of dimensions.
     */
    template <int64_t TDimensions>
    class BaseSurfaces;

    /**
     * @brief Interface for one-dimensional surface geometry.
     */
    using ISurfaces1 = BaseSurfaces<1>;

    /**
     * @brief Interface for two-dimensional surface geometry.
     */
    using ISurfaces2 = BaseSurfaces<2>;

    /**
     * @brief Interface for three-dimensional surface geometry.
     */
    using ISurfaces3 = BaseSurfaces<3>;

    /**
     * @brief Interface for four-dimensional surface geometry.
     */
    using ISurfaces4 = BaseSurfaces<4>;
}
