#pragma once

#include "base_line_fwd.hpp"

#include "iline.hpp"

#include <vislab/core/base_array.hpp>

namespace vislab
{
    /**
     * @brief Base class for line geometry with a certain dimensionality.
     * @tparam TD Dummy parameter needed to make this type reflection compatible.
     * @tparam TDimensions Number of dimensions.
     */
    template <int64_t TDimensions>
    class BaseLine : public Interface<BaseLine<TDimensions>, ILine>
    {
    public:
        /**
         * @brief Number of spatial dimensions.
         */
        static const int64_t Dimensions = TDimensions;

        /**
         * @brief Bounding box type for this geometry.
         */
        using BoundingBox = Eigen::AlignedBox<double, TDimensions>;

        /**
         * @brief Constructor.
         */
        BaseLine()
        {
            mBoundingBox.setEmpty();
        }

        /**
         * @brief Copy-constructor.
         * @param other Geometry to copy from.
         */
        BaseLine(const BaseLine& other)
            : Interface<BaseLine<TDimensions>, ILine>(other)
        {
            mBoundingBox = other.mBoundingBox;
        }

        /**
         * @brief Gets the vertex position array.
         * @return Vertex position array.
         */
        [[nodiscard]] inline std::shared_ptr<BaseArray<TDimensions>> getVertices() { return std::static_pointer_cast<BaseArray<TDimensions>>(this->getVerticesImpl()); }

        /**
         * @brief Gets the vertex position array.
         * @return Vertex position array.
         */
        [[nodiscard]] inline std::shared_ptr<const BaseArray<TDimensions>> getVertices() const { return std::static_pointer_cast<const BaseArray<TDimensions>>(this->getVerticesImpl()); }

        /**
         * @brief Gets the bounding box of the vertices. Note that recomputeBoundingBox has to be called first!
         * @return Bounding box of this geometry.
         */
        [[nodiscard]] inline const BoundingBox& getBoundingBox() const { return mBoundingBox; }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override
        {
            ILine::serialize(archive);
            archive("BoundingBox", mBoundingBox);
        }

        /**
         * @brief Tests if two point geometries are equal.
         * @param other Points to compare to.
         * @return True, if the geometries are equal.
         */
        [[nodiscard]] bool isEqual(const ILine* other) const override
        {
            if (!ILine::isEqual(other))
                return false;

            const BaseLine* otherTyped = dynamic_cast<const BaseLine*>(other);
            if (otherTyped == nullptr)
                return false;

            if (!mBoundingBox.isApprox(otherTyped->mBoundingBox))
                return false;

            return true;
        }

        /**
         * @brief Removes all vertices and attributes.
         */
        void clear() override
        {
            ILine::clear();
            mBoundingBox.setEmpty();
        }

    protected:
        /**
         * @brief Bounding box of the vertices.
         */
        BoundingBox mBoundingBox;
    };
}
