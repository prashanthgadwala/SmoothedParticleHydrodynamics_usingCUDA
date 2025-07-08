#pragma once

#include <stdint.h>

namespace vislab
{
    /**
     * @brief Base class for set of line geometries with a certain dimensionality.
     * @tparam TDimensions Number of dimensions.
     */
    template <int64_t TDimensions>
    class BaseLines;

    /**
     * @brief Interface for one-dimensional line geometry.
     */
    using ILines1 = BaseLines<1>;

    /**
     * @brief Interface for two-dimensional line geometry.
     */
    using ILines2 = BaseLines<2>;

    /**
     * @brief Interface for three-dimensional line geometry.
     */
    using ILines3 = BaseLines<3>;

    /**
     * @brief Interface for four-dimensional line geometry.
     */
    using ILines4 = BaseLines<4>;
}
