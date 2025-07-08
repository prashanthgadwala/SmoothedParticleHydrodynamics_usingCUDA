#pragma once

#include <stdint.h>

namespace vislab
{
    /**
     * @brief Interface for grid data structures with a certain dimension.
     * @tparam TDimensions Total number of dimensions.
     */
    template <int64_t TDimensions>
    class BaseGrid;

    /**
     * @brief Base class for grid data structures in one dimension.
     */
    using IGrid1 = BaseGrid<1>;

    /**
     * @brief Base class for grid data structures in two dimensions.
     */
    using IGrid2 = BaseGrid<2>;

    /**
     * @brief Base class for grid data structures in three dimensions.
     */
    using IGrid3 = BaseGrid<3>;

    /**
     * @brief Base class for grid data structures in four dimensions.
     */
    using IGrid4 = BaseGrid<4>;
}
