#pragma once

#include "component.hpp"

#include <Eigen/Eigen>

namespace vislab
{
    /**
     * @brief Base class for a geometric object.
     */
    class Geometry : public Interface<Geometry, Component>
    {
    public:
        /**
         * @brief Constructor.
        */
        Geometry();

        /**
         * @brief Bounding box of the object in object-space.
         * @return Object-space bounding box.
         */
        [[nodiscard]] virtual Eigen::AlignedBox3d objectBounds() const = 0;

        /**
         * @brief Reverses the orientation of the normals.
         */
        bool reverseOrientation;

        /**
         * @brief Flag that orientates the shading normal during surface interaction computation towards the incoming ray direction. If the flag is disabled, the orientation of the normal is determined by the primitive winding order.
        */
        bool doubleSidedShading;
    };
}
