#pragma once

#include <vislab/core/data.hpp>

namespace vislab
{
    /**
     * @brief Basic interface for grid data structures in any dimension.
     */
    class IGrid : public Interface<IGrid, Data>
    {
    public:
        /**
         * @brief Gets the total number of grid points.
         * @return Number of grid points.
         */
        [[nodiscard]] virtual Eigen::Index getNumGridPoints() const = 0;

        /**
         * @brief Gets the number of cells.
         * @return Number of cells.
         */
        [[nodiscard]] virtual Eigen::Index getNumCells() const = 0;

        /**
         * @brief Gets the number of dimensions.
         * @return Number of dimensions.
         */
        [[nodiscard]] virtual Eigen::Index getNumDimensions() const = 0;

        /**
         * @brief Gets the linear grid point indices of a given cell.
         * @param cellIndex Index of cell to get.
         * @return Grid point indices.
         */
        [[nodiscard]] virtual Eigen::VectorX<Eigen::Index> getCell(Eigen::Index cellIndex) const = 0;
    };
}
