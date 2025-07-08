#pragma once

#include "geometry.hpp"

#include <vislab/geometry/surfaces_fwd.hpp>

namespace vislab
{
    /**
     * @brief Class that contains a collection of triangle meshes using a surface geometry.
     */
    class TrimeshGeometry : public Concrete<TrimeshGeometry, Geometry>
    {
    public:
        /**
         * @brief Constructor.
         */
        TrimeshGeometry();

        /**
         * @brief Constructor with an initial set of surfaces.
         * @param surfaces Surfaces to initalize the triangle mesh with.
         */
        TrimeshGeometry(std::shared_ptr<const Surfaces3f> surfaces);

        /**
         * @brief Bounding box of the object in object-space.
         * @return Object-space bounding box.
         */
        [[nodiscard]] Eigen::AlignedBox3d objectBounds() const override;

        /**
         * @brief Recomputes the bounding box from the vertex buffers.
         */
        void recomputeBoundingBox();

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Vertex positions in object space.
         */
        std::shared_ptr<Array3f> positions;

        /**
         * @brief Vertex normals in object space.
         */
        std::shared_ptr<Array3f> normals;

        /**
         * @brief Texture coordinates.
         */
        std::shared_ptr<Array2f> texCoords;

        /**
         * @brief Data attribute that can for example be mapped to color.
         */
        std::shared_ptr<Array1f> data;

        /**
         * @brief Index buffer containing the indices of triangles in triangle list topology.
         */
        std::shared_ptr<Array3u> indices;

        /**
         * @brief Initializes this object by copying from a surfaces object.
         * @param surfaces 
         */
        void copyFromSurfaces(std::shared_ptr<const Surfaces3f> surfaces);

    private:
        /**
         * @brief Bounding box of the positions in object space.
         */
        Eigen::AlignedBox3d mBoundingBox;
    };
}
