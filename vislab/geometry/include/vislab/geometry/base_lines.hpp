#pragma once

#include "base_lines_fwd.hpp"

#include "base_line.hpp"
#include "ilines.hpp"

#include <vislab/core/base_array.hpp>

namespace vislab
{
    /**
     * @brief Base class for line geometry with a certain dimensionality.
     * @tparam TD Dummy parameter needed to make this type reflection compatible.
     * @tparam TDimensions Number of dimensions.
     */
    template <int64_t TDimensions>
    class BaseLines : public Interface<BaseLines<TDimensions>, ILines>
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
        BaseLines()
        {
            mBoundingBox.setEmpty();
        }

        /**
         * @brief Gets a line.
         * @param index Index of the line to get.
         * @return Line to get.
         */
        [[nodiscard]] std::shared_ptr<BaseLine<TDimensions>> getLine(std::size_t index)
        {
            return std::dynamic_pointer_cast<BaseLine<TDimensions>>(this->getLineImpl(index));
        }

        /**
         * @brief Gets a line.
         * @param index Index of the line to get.
         * @return Line to get.
         */
        [[nodiscard]] std::shared_ptr<const BaseLine<TDimensions>> getLine(std::size_t index) const
        {
            return std::dynamic_pointer_cast<const BaseLine<TDimensions>>(this->getLineImpl(index));
        }

        /**
         * @brief Adds a line that matches the type of the derived class.
         * @param line Line to add.
         */
        void addLine(std::shared_ptr<BaseLine<TDimensions>> line)
        {
            this->addLineImpl(line);
        }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override
        {
            archive("BoundingBox", mBoundingBox);
        }

        /**
         * @brief Gets the bounding box of the vertices. Note that recomputeBoundingBox has to be called first!
         * @return Bounding box of this geometry.
         */
        [[nodiscard]] inline const BoundingBox& getBoundingBox() const { return mBoundingBox; }

        /**
         * @brief Recomputes the bounding box from the vertex buffers.
         */
        void recomputeBoundingBox() override
        {
            mBoundingBox.setEmpty();
            for (std::size_t i = 0; i < this->getNumLines(); ++i)
            {
                auto line = getLine(i);
                line->recomputeBoundingBox();
                mBoundingBox.extend(line->getBoundingBox());
            }
        }

        /**
         * @brief Tests if two line geometries are equal.
         * @param other Line set to compare with.
         * @return True, if the line sets are equal.
         */
        [[nodiscard]] bool isEqual(const ILines* other) const override
        {
            auto otherTyped = dynamic_cast<const BaseLines<Dimensions>*>(other);
            if (otherTyped == nullptr)
                return false;
            if (!mBoundingBox.isApprox(otherTyped->mBoundingBox))
                return false;
            return true;
        }

    protected:
        /**
         * @brief Bounding box of the vertices.
         */
        BoundingBox mBoundingBox;
    };
}
