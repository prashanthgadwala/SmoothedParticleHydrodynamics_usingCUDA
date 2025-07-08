#pragma once

#include <stdint.h>

namespace vislab
{
    /**
     * @brief Base class for line geometry with a certain dimensionality.
     * @tparam TDimensions Number of dimensions.
     */
    template <int64_t TDimensions>
    class BaseLine;

    /**
     * @brief Interface for one-dimensional line.
     */
    using ILine1 = BaseLine<1>;

    /**
     * @brief Interface for two-dimensional line.
     */
    using ILine2 = BaseLine<2>;

    /**
     * @brief Interface for three-dimensional line.
     */
    using ILine3 = BaseLine<3>;

    /**
     * @brief Interface for four-dimensional line.
     */
    using ILine4 = BaseLine<4>;
}
