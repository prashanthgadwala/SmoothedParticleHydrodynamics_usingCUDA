#pragma once

#include "base_grid_fwd.hpp"

#include "igrid.hpp"

namespace vislab
{
    /**
     * @brief Interface for grid data structures with a certain dimension.
     * @tparam TDimensions Total number of dimensions.
     */
    template <int64_t TDimensions>
    class BaseGrid : public Interface<BaseGrid<TDimensions>, IGrid>
    {
    public:
        /**
         * @brief Gets the number of dimensions.
         */
        static const int64_t Dimensions = TDimensions;

        /**
         * @brief Gets the number of dimensions.
         * @return Number of dimensions.
         */
        [[nodiscard]] inline Eigen::Index getNumDimensions() const override
        {
            return Dimensions;
        }
    };
}
