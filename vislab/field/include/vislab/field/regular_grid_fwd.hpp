#pragma once

#include <stdint.h>

namespace vislab
{
    /**
     * @brief Basic interface for regular grids in a certain dimension.
     * @tparam TScalar Scalar value used to store data on the grid.
     * @tparam TDimensions Number of dimensions.
     */
    template <typename TScalar, int64_t TDimensions>
    class RegularGrid;

    /**
     * @brief One-dimensional regular grid.
     */
    using RegularGrid1d = RegularGrid<double, 1>;

    /**
     * @brief Two-dimensional regular grid.
     */
    using RegularGrid2d = RegularGrid<double, 2>;

    /**
     * @brief Three-dimensional regular grid.
     */
    using RegularGrid3d = RegularGrid<double, 3>;

    /**
     * @brief Four-dimensional regular grid.
     */
    using RegularGrid4d = RegularGrid<double, 4>;
}
