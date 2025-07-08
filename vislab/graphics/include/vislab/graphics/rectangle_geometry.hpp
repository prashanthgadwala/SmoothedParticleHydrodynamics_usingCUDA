#pragma once

#include "geometry.hpp"

namespace vislab
{
    /**
     * @brief Class that defines a rectangle spanning (-1,-1,0) to (1,1,0). To modify its position update the transformation.
     */
    class RectangleGeometry : public Concrete<RectangleGeometry, Geometry>
    {
    public:
        /**
         * @brief Bounding box of the object in object-space.
         * @return Object-space bounding box.
         */
        [[nodiscard]] Eigen::AlignedBox3d objectBounds() const override;

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;
    };
}
