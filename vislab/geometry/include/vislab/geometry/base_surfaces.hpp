#pragma once

#include "base_surfaces_fwd.hpp"

#include "base_surface.hpp"
#include "isurfaces.hpp"

#include <vislab/core/base_array.hpp>

namespace vislab
{
    /**
     * @brief Base class for point geometry with a certain dimensionality.
     * @tparam TDimensions Number of dimensions.
     */
    template <int64_t TDimensions>
    class BaseSurfaces : public Interface<BaseSurfaces<TDimensions>, ISurfaces>
    {
    public:
        /**
         * @brief Number of spatial dimensions.
         */
        static constexpr int Dimensions = TDimensions;

        /**
         * @brief Bounding box type for this geometry.
         */
        using BoundingBox = Eigen::AlignedBox<double, TDimensions>;

        /**
         * @brief Constructor.
         */
        BaseSurfaces()
        {
            mBoundingBox.setEmpty();
        }

        /**
         * @brief Gets a surface.
         * @param index Index of the surface to get.
         * @return Surface to get.
         */
        [[nodiscard]] inline std::shared_ptr<BaseSurface<TDimensions>> getSurface(std::size_t index)
        {
            return std::dynamic_pointer_cast<BaseSurface<TDimensions>>(this->getSurfaceImpl(index));
        }

        /**
         * @brief Gets a surface.
         * @param index Index of the surface to get.
         * @return Surface to get.
         */
        [[nodiscard]] inline std::shared_ptr<const BaseSurface<TDimensions>> getSurface(std::size_t index) const
        {
            return std::dynamic_pointer_cast<const BaseSurface<TDimensions>>(this->getSurfaceImpl(index));
        }

        /**
         * @brief Adds a surface that matches the type of the derived class.
         * @param surface Surface to add.
         */
        void addSurface(std::shared_ptr<BaseSurface<TDimensions>> surface)
        {
            this->addSurfaceImpl(surface);
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
            for (std::size_t i = 0; i < this->getNumSurfaces(); ++i)
            {
                auto surface = getSurface(i);
                surface->recomputeBoundingBox();
                mBoundingBox.extend(surface->getBoundingBox());
            }
        }

        /**
         * @brief Tests if two surface geometries are equal.
         * @param other Surface set to compare with.
         * @return True, if the surface sets are equal.
         */
        [[nodiscard]] bool isEqual(const ISurfaces* other) const override
        {
            auto otherTyped = dynamic_cast<const BaseSurfaces<Dimensions>*>(other);
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
