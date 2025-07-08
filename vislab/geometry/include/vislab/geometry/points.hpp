#pragma once

#include "points_fwd.hpp"

#include "attributes.hpp"
#include "base_points.hpp"

#include <vislab/core/array.hpp>
#include <vislab/core/iarchive.hpp>
#include <vislab/core/traits.hpp>

namespace vislab
{
    /**
     * @brief Point geometry with specific internal type.
     * @tparam TArrayType Internal array type used to store vertices.
     */
    template <typename TArrayType>
    class Points : public Concrete<Points<TArrayType>, BasePoints<TArrayType::Dimensions>>
    {
    public:
        /**
         * @brief Number of dimensions for vertex positions.
         */
        static constexpr int Dimensions = TArrayType::Dimensions;

        /**
         * @brief Internal array type used to store the vertices.
         */
        using ArrayType = TArrayType;

        /**
         * @brief Constructor.
         */
        Points()
        {
            vertices = std::make_shared<TArrayType>();
        }

        /**
         * @brief Copy-constructor.
         * @param other Geometry to copy from.
         */
        Points(const Points& other)
            : Concrete<Points<TArrayType>, BasePoints<TArrayType::Dimensions>>(other)
        {
            vertices = other.vertices->clone();
        }

        /**
         * @brief Gets the vertex position array.
         * @return Vertex position array.
         */
        [[nodiscard]] inline std::shared_ptr<TArrayType> getVertices() { return vertices; }

        /**
         * @brief Gets the vertex position array.
         * @return Vertex position array.
         */
        [[nodiscard]] inline std::shared_ptr<const TArrayType> getVertices() const { return vertices; }

        /**
         * @brief Recomputes the bounding box from the vertex buffer.
         */
        void recomputeBoundingBox() override
        {
            this->mBoundingBox.setEmpty();
            if (vertices->getSize() == 0)
                return;
            this->mBoundingBox.extend(vertices->getMin().template cast<double>());
            this->mBoundingBox.extend(vertices->getMax().template cast<double>());
        }

        /**
         * @brief Checks if the points are properly initialized.
         * @return True if the bounding box is valid and if attributes are defined for each vertex (if they are defined).
         */
        bool isValid() const override
        {
            // attributes deleted or invalid?
            if (!this->attributes || !this->attributes->isValid())
                return false;

            // vertices deleted?
            if (!vertices)
                return false;

            // bounding box computed?
            if (vertices->getSize() != 0 && this->mBoundingBox.isEmpty())
                return false;

            // attributes defined per vertex?
            for (std::size_t iattr = 0; iattr < this->attributes->getSize(); ++iattr)
                if (this->attributes->getByIndex(iattr)->getSize() != vertices->getSize())
                    return false;

            // all good
            return true;
        }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override
        {
            BasePoints<TArrayType::Dimensions>::serialize(archive);
            archive("Vertices", vertices);
        }

        /**
         * @brief Array that stores the vertex positions.
         */
        std::shared_ptr<TArrayType> vertices;

    private:
        /**
         * @brief Gets the vertex position array.
         * @return Vertex position array.
         */
        [[nodiscard]] inline std::shared_ptr<IArray> getVerticesImpl() override { return vertices; }

        /**
         * @brief Gets the vertex position array.
         * @return Vertex position array.
         */
        [[nodiscard]] inline std::shared_ptr<const IArray> getVerticesImpl() const override { return vertices; }
    };
}
