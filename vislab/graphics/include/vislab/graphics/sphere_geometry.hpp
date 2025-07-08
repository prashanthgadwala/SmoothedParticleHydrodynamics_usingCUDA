#pragma once

#include "geometry.hpp"

#include <vislab/core/array_fwd.hpp>

namespace vislab
{
    /**
     * @brief Class that contains a collection of spheres.
     */
    class SphereGeometry : public Concrete<SphereGeometry, Geometry>
    {
    public:
        /**
         * @brief Constructor.
         * @param radiusScale Scaling factor that changes the radius of the sphere.
         */
        SphereGeometry(float radiusScale = 1);

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

        /**
         * @brief Positions of the spheres in object space.
         */
        std::shared_ptr<Array3f> positions;

        /**
         * @brief Linear scaling factor of the radius.
         */
        float radiusScale;

        /**
         * @brief Sphere radius in object space.
         */
        std::shared_ptr<Array1f> radius;

        /**
         * @brief Data attribute that can for example be mapped to color.
         */
        std::shared_ptr<Array1f> data;
    };
}
