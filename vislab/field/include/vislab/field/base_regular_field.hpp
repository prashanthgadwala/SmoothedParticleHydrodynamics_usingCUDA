#pragma once

#include "base_regular_field_fwd.hpp"

#include "regular_grid.hpp"

#include <vislab/core/array.hpp>

namespace vislab
{
    /**
     * @brief Enumeration of boundary behaviors supported by the regular field.
     */
    enum class EBoundaryBehavior
    {
        /**
         * @brief Samples outside the domain get clamped to the interior.
         */
        Clamp,

        /**
         * @brief The domain is periodic and samples that move out on one side are consided to be coming in from the opposite side.
         */
        Periodic
    };

    /**
     * @brief Base class for fields defined on regular grids.
     * @tparam TBaseType Field type to discretize onto a regular grid.
     * @tparam TGridType Grid type to discretize on.
     */
    template <class TBaseType, typename TGridType>
    class BaseRegularField : public Interface<BaseRegularField<TBaseType, TGridType>, TBaseType>
    {
    public:
        /**
         * @brief Number of input dimensions.
         */
        static constexpr int Dimensions = TBaseType::Dimensions;

        /**
         * @brief Number of output dimensions.
         */
        static constexpr int Components = TBaseType::Components;

        /**
         * @brief Type of values in the field.
         */
        using Value = typename TBaseType::Value;

        /**
         * @brief Type of the domain coordinates.
         */
        using DomainCoord = typename TBaseType::DomainCoord;

        /**
         * @brief Type to specify a partial derivative. This type is used in the samplePartial function to specify the desired derivative of each dimension.
         */
        using Partial = typename TBaseType::Partial;

        /**
         * @brief Type of the underlying regular grid
         */
        using GridType = TGridType;

        /**
         * @brief Type of the underlying regular grid
         */
        using ArrayType = BaseArray<Components>;

        /**
         * @brief Type of the bounding box.
         */
        using BoundingBox = Eigen::AlignedBox<double, Dimensions>;
        
        /**
         * @brief Default constructor.
         */
        BaseRegularField()
            : Interface<BaseRegularField<TBaseType, TGridType>, TBaseType>(BoundingBox())
            , mAccuracy(2)
        {
            for (int i = 0; i < Dimensions; ++i)
                mBoundaryBehavior[i] = EBoundaryBehavior::Clamp;
        }

        /**
         * @brief Copy constructor.
         * @param other RegularField to copy from.
         */
        BaseRegularField(const BaseRegularField& other)
            : Interface<BaseRegularField<TBaseType, TGridType>, TBaseType>(BoundingBox())
            , mBoundaryBehavior(other.mBoundaryBehavior)
            , mAccuracy(other.mAccuracy)
        {
            if (other.mGrid != nullptr)
                mGrid = other.mGrid->clone();
        }

        /**
         * @brief Sets the grid.
         * @param grid Grid to set.
         */
        void setGrid(std::shared_ptr<GridType> grid) { mGrid = grid; }

        /**
         * @brief Gets the grid.
         * @return Underlying grid.
         */
        [[nodiscard]] inline std::shared_ptr<const GridType> getGrid() const { return mGrid; }
        
        /**
         * @brief Gets the grid.
         * @return Underlying grid.
         */
        [[nodiscard]] inline std::shared_ptr<GridType> getGrid() { return mGrid; }

        /**
         * @brief Gets the data array.
         * @return Underlying array that contains the data.
         */
        [[nodiscard]] inline std::shared_ptr<const ArrayType> getArray() const { return getArrayImpl(); }

        /**
         * @brief Gets the data array.
         * @return Underlying array that contains the data.
         */
        [[nodiscard]] inline std::shared_ptr<ArrayType> getArray() { return getArrayImpl(); }

        /**
         * @brief Gets the domain of the field.
         * @return Domain bounding box.
         */
        [[nodiscard]] inline const BoundingBox& getDomain() const override { return mGrid->getDomain(); }

        /**
         * @brief Sets the domain of the field.
         * @param domain Domain bounding box to set.
         */
        void setDomain(const BoundingBox& domain) override
        {
            this->mDomain = domain;
            mGrid->setDomain(domain);
        }

        /**
         * @brief Gets the boundary behavior for a certain dimension.
         * @param dimension Dimension to get boundary behavior for.
         * @return Boundary behavior of requested dimension.
         */
        [[nodiscard]] inline EBoundaryBehavior getBoundaryBehavior(int dimension) const
        {
            return mBoundaryBehavior[dimension];
        }

        /**
         * @brief Sets the boundary behavior for a certain dimension.
         * @param dimension Dimension to set boundary behavior for.
         * @param behavior Boundary behavior to set.
         */
        inline void setBoundaryBehavior(int dimension, EBoundaryBehavior behavior)
        {
            mBoundaryBehavior[dimension] = behavior;
        }

        /**
         * @brief Gets the accuracy of the Newton interpolation. This parameter determines the degree of the interpolation polynomial.
         * @return Accuracy of Newton interpolation.
         */
        [[nodiscard]] inline int getAccuracy() const
        {
            return mAccuracy;
        }

        /**
         * @brief Sets the accuracy of the Newton interpolation. This parameter determines the degree of the interpolation polynomial.
         * @param accuracy Accuracy of Newton interpolation to set.
         */
        inline void setAccuracy(int accuracy)
        {
            mAccuracy = accuracy;
        }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override
        {
            TBaseType::serialize(archive);
            archive("Grid", mGrid);
            std::array<int, Dimensions> copy;
            for (int i = 0; i < Dimensions; ++i)
                copy[i] = static_cast<int>(mBoundaryBehavior[i]);
            archive("BoundaryBehavior", copy);
            for (int i = 0; i < Dimensions; ++i)
                mBoundaryBehavior[i] = static_cast<EBoundaryBehavior>(copy[i]);
            archive("Accuracy", mAccuracy);
        }

    protected:
        /**
         * @brief Gets the data array.
         * @return Underlying array that contains the data.
         */
        [[nodiscard]] virtual std::shared_ptr<ArrayType> getArrayImpl() = 0;

        /**
         * @brief Gets the data array.
         * @return Underlying array that contains the data.
         */
        [[nodiscard]] virtual std::shared_ptr<const ArrayType> getArrayImpl() const = 0;

        /**
         * @brief Grid on which the data is stored.
         */
        std::shared_ptr<GridType> mGrid;

        /**
         * @brief Sets the behavior on the axis-aligned boundaries for samples that land outside of the domain.
         */
        std::array<EBoundaryBehavior, Dimensions> mBoundaryBehavior;

        /**
         * @brief Accuracy of the Newton interpolation. This parameter determines the degree of the interpolation polynomial.
         */
        int mAccuracy;
    };
}
