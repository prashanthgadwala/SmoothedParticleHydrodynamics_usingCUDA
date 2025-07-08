#pragma once

#include "surface_fwd.hpp"

#include "base_surface.hpp"

#include "attributes.hpp"
#include "primitive_topology.hpp"

#include <vislab/core/array.hpp>
#include <vislab/core/iarchive.hpp>
#include <vislab/core/traits.hpp>

namespace vislab
{
    /**
     * @brief Surface geometry with specific internal type.
     * @tparam TArrayType Internal array type used to store vertices.
     */
    template <typename TVertexArrayType, typename TIndexArrayType>
    class Surface : public Concrete<Surface<TVertexArrayType, TIndexArrayType>, BaseSurface<TVertexArrayType::Dimensions>>
    {
    public:
        /**
         * @brief Number of dimensions for vertex positions.
         */
        static constexpr int Dimensions = TVertexArrayType::Dimensions;

        /**
         * @brief Internal array type used to store positions.
         */
        using PositionArrayType = TVertexArrayType;

        /**
         * @brief Internal array type used to store normals.
         */
        using NormalArrayType = TVertexArrayType;

        /**
         * @brief Internal array type used to store texture coordinates.
         */
        using TexCoordArrayType = Array<Eigen::Vector<typename TVertexArrayType::Scalar, 2>>;

        /**
         * @brief Internal array type used to store the indices.
         */
        using IndexArrayType = TIndexArrayType;

        /**
         * @brief Constructor.
         */
        Surface()
            : primitiveTopology(EPrimitiveTopology::TriangleList)
            , positions(std::make_shared<PositionArrayType>())
            , normals(nullptr)
            , texCoords(nullptr)
            , indices(std::make_shared<TIndexArrayType>())
        {
        }

        /**
         * @brief Copy-constructor.
         * @param other Geometry to copy from.
         */
        Surface(const Surface& other)
            : Concrete<Surface<TVertexArrayType, TIndexArrayType>, BaseSurface<TVertexArrayType::Dimensions>>(other)
            , primitiveTopology(other.primitiveTopology)
        {
            positions = other.positions->clone();
            if (other.normals)
                normals = other.normals->clone();
            if (other.texCoords)
                texCoords = other.texCoords->clone();
            indices = other.indices->clone();
        }

        /**
         * @brief Gets the vertex position array.
         * @return Vertex position array.
         */
        [[nodiscard]] inline std::shared_ptr<PositionArrayType> getPositions() { return positions; }

        /**
         * @brief Gets the vertex position array.
         * @return Vertex position array.
         */
        [[nodiscard]] inline std::shared_ptr<const PositionArrayType> getPositions() const { return positions; }

        /**
         * @brief Recomputes the bounding box from the vertex buffer.
         */
        void recomputeBoundingBox() override
        {
            this->mBoundingBox.setEmpty();
            if (positions->getSize() == 0)
                return;

            this->mBoundingBox.extend(positions->getMin().template cast<double>());
            this->mBoundingBox.extend(positions->getMax().template cast<double>());
        }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override
        {
            BaseSurface<TVertexArrayType::Dimensions>::serialize(archive);
            archive("Positions", positions);
            archive("Normals", normals);
            archive("TexCoords", texCoords);
            archive("Indices", indices);

            int topo = (int)primitiveTopology;
            archive("PrimitiveTopology", topo);
            primitiveTopology = (EPrimitiveTopology)topo;
        }

        /**
         * @brief Tests if two surface geometries are equal.
         * @param other Surface to compare to.
         * @return True, if the geometries are equal.
         */
        [[nodiscard]] bool isEqual(const ISurface* other) const override
        {
            if (!BaseSurface<TVertexArrayType::Dimensions>::isEqual(other))
                return false;

            const Surface* otherTyped = dynamic_cast<const Surface*>(other);
            if (otherTyped == nullptr)
                return false;

            if (!positions->isEqual(otherTyped->positions.get()))
                return false;

            if ((normals && !otherTyped->normals) || (!normals && otherTyped->normals))
                return false;

            if (otherTyped->normals && !normals->isEqual(otherTyped->normals.get()))
                return false;

            if ((texCoords && !otherTyped->texCoords) || (!texCoords && otherTyped->texCoords))
                return false;

            if (otherTyped->texCoords && !texCoords->isEqual(otherTyped->texCoords.get()))
                return false;

            if (!indices->isEqual(otherTyped->indices.get()))
                return false;

            return true;
        }

        /**
         * @brief Removes all vertices and attributes.
         */
        void clear() override
        {
            BaseSurface<TVertexArrayType::Dimensions>::clear();
            indices->clear();
            positions->clear();
            normals->clear();
            texCoords->clear();
        }

        /**
         * @brief Checks if the line is properly initialized.
         * @return True if the bounding box is valid and if attributes are defined for each vertex (if they are defined).
         */
        bool isValid() const override
        {
            // attributes deleted or invalid?
            if (!this->attributes || !this->attributes->isValid())
                return false;

            // positions or indices deleted?
            if (!positions || !indices)
                return false;

            // are there indices?
            if (positions->getSize() != 0 && indices->getSize() == 0)
                return false;

            // bounding box computed?
            if (positions->getSize() != 0 && this->mBoundingBox.isEmpty())
                return false;

            // attributes defined per vertex?
            for (std::size_t iattr = 0; iattr < this->attributes->getSize(); ++iattr)
                if (this->attributes->getByIndex(iattr)->getSize() != positions->getSize())
                    return false;

            // normal defined per vertex? (if there are normals)
            if (normals && normals->getSize() > 0 && (positions->getSize() != normals->getSize()))
                return false;

            // texCoords defined per vertex? (if there are texCoords)
            if (texCoords && normals->getSize() > 0 && (positions->getSize() != normals->getSize()))
                return false;

            // all good
            return true;
        }

        /**
         * @brief Array that stores the vertex positions.
         */
        std::shared_ptr<PositionArrayType> positions;

        /**
         * @brief Array that stores the vertex normals.
         */
        std::shared_ptr<NormalArrayType> normals;

        /**
         * @brief Array that stores the vertex texture coordinates.
         */
        std::shared_ptr<TexCoordArrayType> texCoords;

        /**
         * @brief Primitive topology.
         */
        EPrimitiveTopology primitiveTopology;

        /**
         * @brief Index buffer.
         */
        std::shared_ptr<TIndexArrayType> indices;

    private:
        /**
         * @brief Gets the vertex position array.
         * @return Vertex position array.
         */
        [[nodiscard]] inline std::shared_ptr<IArray> getPositionsImpl() override { return positions; }

        /**
         * @brief Gets the vertex position array.
         * @return Vertex position array.
         */
        [[nodiscard]] inline std::shared_ptr<const IArray> getPositionsImpl() const override { return positions; }
    };
}
