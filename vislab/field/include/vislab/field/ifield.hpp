#pragma once

#include <vislab/core/data.hpp>

#include <Eigen/Eigen>

namespace vislab
{
    /**
     * @brief Basic interface for any type of field.
     */
    class IField : public Interface<IField, Data>
    {
    public:
        /**
         * @brief Gets the number of input dimensions to this field (space + time).
         * @return Number of dimensions.
         */
        [[nodiscard]] virtual Eigen::Index getDimensions() const = 0;

        /**
         * @brief Gets the number of input dimensions to this field (space only).
         * @return Number of spatial dimensions.
         */
        [[nodiscard]] virtual Eigen::Index getSpatialDimensions() const = 0;
    };
}
