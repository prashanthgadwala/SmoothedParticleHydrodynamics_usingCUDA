#pragma once

#include <stdint.h>

namespace vislab
{
    /**
     * @brief Base class for point geometry with a certain dimensionality.
     * @tparam TDimensions Number of dimensions.
     */
    template <int64_t TDimensions>
    class BasePoints;

    /**
     * @brief Interface for one-dimensional point geometry.
     */
    using IPoints1 = BasePoints<1>;

    /**
     * @brief Interface for two-dimensional point geometry.
     */
    using IPoints2 = BasePoints<2>;

    /**
     * @brief Interface for three-dimensional point geometry.
     */
    using IPoints3 = BasePoints<3>;

    /**
     * @brief Interface for four-dimensional point geometry.
     */
    using IPoints4 = BasePoints<4>;
}
