#pragma once

#include "regular_field_fwd.hpp"

#include "iscalar_field.hpp"
#include "ivector_field.hpp"

#include "base_regular_field.hpp"

namespace vislab
{
    /**
     * @brief Base class for fields defined on regular grids.
     * @tparam TBaseType Field type to discretize onto a regular grid.
     * @tparam TGridType Grid type to discretize on.
     */
    template <class TBaseType, typename TGridType, typename TArrayType>
    class RegularField : public Concrete<RegularField<TBaseType, TGridType, TArrayType>, BaseRegularField<TBaseType, TGridType>>
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
         * @brief Number of output rows.
         */
        static constexpr int ComponentRowsAtCompileTime = TBaseType::RowsAtCompileTime;

        /**
         * @brief Number of output columns.
         */
        static constexpr int ComponentColsAtCompileTime = TBaseType::ColsAtCompileTime;

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
        using ArrayType = TArrayType;

        /**
         * @brief Type of values in the underlying array.
         */
        using ArrayElement = typename TArrayType::Element;

        /**
         * @brief Type of the bounding box.
         */
        using BoundingBox = Eigen::AlignedBox<double, Dimensions>;

        /**
         * @brief Default constructor.
         */
        RegularField()
        {
        }

        /**
         * @brief Constructor.
         *
         * @param domain The domain of the field.
         * @param resolution The resolution of the field.
         */
        RegularField(const BoundingBox& domain, const typename GridType::GridCoord& resolution)
        {
            auto grid = std::make_shared<GridType>();
            grid->setResolution(resolution);
            grid->setDomain(domain);
            this->setGrid(grid);

            auto array = std::make_shared<ArrayType>();
            array->setSize(resolution.prod());
            setArray(array);
        }

        /**
         * @brief Copy constructor.
         * @param other RegularField to copy from.
         */
        RegularField(const RegularField& other)
            : Concrete<RegularField<TBaseType, TGridType, TArrayType>, BaseRegularField<TBaseType, TGridType>>(other)
        {
            if (other.mArray != nullptr)
                mArray = other.mArray->clone();
        }

        /**
         * @brief Gets the data array.
         * @return Underlying array that contains the data.
         */
        [[nodiscard]] inline std::shared_ptr<TArrayType> getArray() { return mArray; }

        /**
         * @brief Gets the data array.
         * @return Underlying array that contains the data.
         */
        [[nodiscard]] inline std::shared_ptr<const TArrayType> getArray() const { return mArray; }

        /**
         * @brief Sets the array. (The number of components must match the BaseType!)
         * @param data Array to set.
         */
        inline void setArray(std::shared_ptr<TArrayType> data) { mArray = data; }

        /**
         * @brief Samples the field at an arbitrary location in the domain by employing an interpolation.
         * @param coord Domain location to sample the field at.
         * @return Sampled value.
         */
        [[nodiscard]] inline Value sample(const DomainCoord& coord) const override
        {
            return sampleImpl(coord, Partial::c);
        }

        /**
         * @brief Samples a partial derivative at an arbitrary location in the domain by employing an interpolation.
         * @param coord Domain location to sample the field derivative at.
         * @param partial Specifies the desired partial derivative of each dimension.
         * @return Sampled partial derivative.
         */
        [[nodiscard]] inline Value samplePartial(const DomainCoord& coord, const Partial& partial) const override
        {
            return sampleImpl(coord, partial);
        }

        /**
         * @brief Numerically estimates the first-order partial derivative using finite differences on a grid for a given dimension. For accuracy=2, the coefficients are hard-coded. For anything above, it falls back to samplePartial, which computes the coefficients automatically.
         * @param gridCoord Grid coordinate at which to estimate the finite difference.
         * @param partial Specifies the desired partial derivative of each dimension.
         * @return Estimated partial derivative.
         */
        [[nodiscard]] inline Value estimatePartial(const typename GridType::GridCoord& gridCoord, const Partial& partial) const
        {
            // get the number of non-zeros and figure out which partial to compute
            int nnz   = 0;
            int index = -1;
            for (int d = 0; d < Dimensions; ++d)
            {
                if (partial[d] != 0)
                {
                    nnz++;
                    index = d;
                }
            }

            // call the respective hard-coded functions
            if (nnz == 1)
            {
                if (partial[index] == 1)
                    return estimate_1st_partial(gridCoord, index);
                if (partial[index] == 2)
                    return estimate_2nd_partial(gridCoord, index);
            }

            // fall back to general (but slow) routine.
            return sampleImpl(this->mGrid->getCoordAt(gridCoord), partial);
        }

        /**
         * @brief Gets the vertex data at a specific grid point.
         * @param gridCoord Grid coord to get the value at.
         * @return Value at the grid coord.
         */
        [[nodiscard]] inline ArrayElement getVertexDataAt(const typename GridType::GridCoord& gridCoord) const { return mArray->getValue(this->mGrid->getLinearIndex(gridCoord)); }

        /**
         * @brief Sets the vertex data at a specific grid point.
         * @param gridCoord Grid coord to set the value at.
         * @param value Value to set.
         */
        inline void setVertexDataAt(const typename GridType::GridCoord& gridCoord, const ArrayElement& value) { mArray->setValue(this->mGrid->getLinearIndex(gridCoord), value); }

        /**
         * @brief Sets the vertex data at a specific grid point if it is a scalar value.
         * @tparam D SFINAE parameter.
         * @param gridCoord Grid coord to set the value at.
         * @param value Value to set.
         * @return SFINAE return type.
         */
        template <int D = Components>
        inline typename std::enable_if_t<(D == 1)> setVertexDataAt(const typename GridType::GridCoord& gridCoord, const typename ArrayElement::Scalar& value)
        {
            mArray->setValue(this->mGrid->getLinearIndex(gridCoord), ArrayElement(value));
        }

        /**
         * @brief For each vertex in the grid, execute a function.
         *
         * @param func Function to execute for each vertex.
         */
        inline void forEachVertex(const std::function<void(const typename GridType::GridCoord& gridCoord, ArrayElement& value)>& func)
        {
#ifdef NDEBUG
#pragma omp parallel for
#endif
            for (Eigen::Index i = 0; i < mArray->getSize(); ++i)
            {
                const auto& gridCoord = this->mGrid->getGridCoord(i);
                auto value            = mArray->getValue(i);
                func(gridCoord, value);
                mArray->setValue(i, value);
            }
        }

        /**
         * @brief For each vertex in the grid, execute a function.
         *
         * @param func Function to execute for each vertex.
         */
        inline void forEachVertex(const std::function<void(const typename GridType::GridCoord& gridCoord, const ArrayElement& value)>& func) const
        {
#ifdef NDEBUG
#pragma omp parallel for
#endif
            for (Eigen::Index i = 0; i < mArray->getSize(); ++i)
            {
                const auto& gridCoord = this->mGrid->getGridCoord(i);
                func(gridCoord, mArray->getValue(i));
            }
        }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override
        {
            BaseRegularField<TBaseType, TGridType>::serialize(archive);
            archive("Array", mArray);
        }

        /**
         * @brief Tests if the resolution is set properly (no dimension is zero) and that the domain is not invalid (zero domain). If this function returns false, the object has probably not been initialized correctly with setResolution and setDomain.
         * @return True if the grid is valid.
         */
        [[nodiscard]] bool isValid() const override
        {
            // is the grid valid?
            if (!this->mGrid->isValid())
                return false;

            // is there an output array allocated?
            if (!this->mArray)
                return false;

            // has the output array the correct size?
            if (this->mArray->getSize() != this->mGrid->getResolution().prod())
                return false;

            return true;
        }

    private:
        /**
         * @brief Helper function that implements the sampling of the function and its derivatives.
         * @param coord Physical domain coordinate to sample at.
         * @param partial Specifies the desired derivative of each dimension.
         * @return Function value or its derivative.
         */
        [[nodiscard]] Value sampleImpl(const DomainCoord& coord, const Partial& partial) const
        {
            // get some basic properties about the grid
            auto& minCorner = this->mGrid->getDomain().min();
            auto& maxCorner = this->mGrid->getDomain().max();
            auto& res       = this->mGrid->getResolution();
            auto sep        = this->mGrid->getSpacing();

            // convert requested coordinate to index coordinates
            auto index = ((coord - minCorner).cwiseQuotient(maxCorner - minCorner)).cwiseProduct(res.template cast<double>() - DomainCoord::Ones());

            // calculate the left-most and right-most position to read from (in index coordinates) for each dimension
            Eigen::Vector<int, Dimensions> left, right;
            for (int i = 0; i < Dimensions; ++i)
            {
                int numCoefficients = 2 * ((std::max(uint8_t(1), partial[i]) + 1) / 2) - 1 + this->mAccuracy;

                // on boundary, increase the number of points by one
                if (this->mBoundaryBehavior[i] == EBoundaryBehavior::Clamp && (partial[i] % 2 == 0) && (index[i] == 0 || index[i] == res[i] - 1))
                {
                    numCoefficients += 1;
                }

                right[i] = (int)std::round(index[i]) + numCoefficients / 2;
                switch (this->mBoundaryBehavior[i])
                {
                case EBoundaryBehavior::Periodic:
                    right[i] = right[i] % (res[i] - 1);
                    break;
                case EBoundaryBehavior::Clamp:
                    right[i] = std::min(right[i], res[i] - 1);
                    break;
                }

                left[i] = right[i] - numCoefficients + 1;
                switch (this->mBoundaryBehavior[i])
                {
                case EBoundaryBehavior::Periodic:
                    // left bound can be out-of-range, but this is intentional to keep left<right. periodic handling is done later down below.
                    break;
                case EBoundaryBehavior::Clamp:
                    if (left[i] < 0)
                    {
                        right[i] -= left[i];
                        left[i]  = 0;
                        right[i] = std::min(right[i], res[i] - 1);
                    }
                    break;
                }
            }

            // next, we compute the coefficients per dimension.
            std::vector<double> coefficients[Dimensions];
            Eigen::Index numGridPoints = 1;
            for (int i = 0; i < Dimensions; ++i)
            {
                int numCoefficients = right[i] - left[i] + 1;
                coefficients[i].resize(numCoefficients * (partial[i] + 1));
                calculateWeights(numCoefficients, partial[i], sep[i], coefficients[i].data(), index[i] - left[i]);
                numGridPoints *= numCoefficients;
            }

            // lastly, we loop over the required grid points and compute the weighted sum
            Value result = Value::Zero();
            for (Eigen::Index localLinearIndex = 0; localLinearIndex < numGridPoints; ++localLinearIndex)
            {
                // compute the grid index in the stencil window
                typename GridType::GridCoord localGridIndex;
                Eigen::Index stride = 1;
                for (int d = 0; d < Dimensions - 1; ++d)
                    stride *= right[d] - left[d] + 1;
                Eigen::Index t = localLinearIndex;
                for (int64_t d = Dimensions - 1; d >= 0; --d)
                {
                    localGridIndex[d] = (int)(t / stride);
                    t                 = t % stride;
                    if (d > 0)
                        stride /= right[d - 1] - left[d - 1] + 1;
                }

                // compute the grid index in the full grid
                typename GridType::GridCoord gridIndex = localGridIndex + left;

                // do periodic handling for left bound
                for (int i = 0; i < Dimensions; ++i)
                {
                    switch (this->mBoundaryBehavior[i])
                    {
                    case EBoundaryBehavior::Periodic:
                        gridIndex[i] = gridIndex[i] % (res[i] - 1);
                        break;
                    case EBoundaryBehavior::Clamp:
                        break;
                    }
                }

                // compute the product of the weights
                double weight = 1;
                for (int i = 0; i < Dimensions; ++i)
                {
                    // we are offseting the coefficients, since the coefficients store the weights for the lower derivatives, as well.
                    weight *= coefficients[i][(right[i] - left[i] + 1) * partial[i] + localGridIndex[i]];
                }

                // compute contribution from this stencil pixel and add to sum.
                Eigen::Index linearIndex = this->mGrid->getLinearIndex(gridIndex);
                result += mArray->getValue(linearIndex).template cast<double>() * weight;
            }

            // return the result
            return result;
        }

        /**
         * @brief Numerically estimates estimate_1st_partialthe first-order partial derivative using finite differences on a grid for a given dimension. For accuracy=2, the coefficients are hard-coded. For anything above, it falls back to sample_dx, which computes the coefficients automatically.
         * @param gridCoord Grid coordinate at which to estimate the finite difference.
         * @return Estimated partial derivative.
         */
        [[nodiscard]] Value estimate_1st_partial(const typename GridType::GridCoord& gridCoord, int dimension) const
        {
            double spacing             = this->mGrid->getSpacing()[dimension];
            auto resolution            = this->mGrid->getResolution();
            Eigen::Index linearIndex   = this->mGrid->getLinearIndex(gridCoord);
            auto strides               = this->mGrid->getPointStrides();
            EBoundaryBehavior boundary = this->mBoundaryBehavior[dimension];

            switch (this->mAccuracy)
            {
            case 2:
                if (gridCoord[dimension] > 0 && gridCoord[dimension] < resolution[dimension] - 1 && resolution[dimension] >= 2 && boundary == EBoundaryBehavior::Clamp)
                {
                    return (-0.5 * mArray->getValue(linearIndex - strides[dimension]).template cast<double>() +
                            +0.5 * mArray->getValue(linearIndex + strides[dimension]).template cast<double>()) /
                           spacing;
                }
                else if (gridCoord[dimension] == 0 && resolution[dimension] >= 2 && boundary == EBoundaryBehavior::Clamp)
                {
                    return (-1.5 * mArray->getValue(linearIndex).template cast<double>() +
                            +2.0 * mArray->getValue(linearIndex + strides[dimension]).template cast<double>() +
                            -0.5 * mArray->getValue(linearIndex + 2 * strides[dimension]).template cast<double>()) /
                           spacing;
                }
                else if (gridCoord[dimension] == resolution[dimension] - 1 && resolution[dimension] >= 2 && boundary == EBoundaryBehavior::Clamp)
                {
                    return (+0.5 * mArray->getValue(linearIndex - 2 * strides[dimension]).template cast<double>() +
                            -2.0 * mArray->getValue(linearIndex - strides[dimension]).template cast<double>() +
                            +1.5 * mArray->getValue(linearIndex).template cast<double>()) /
                           spacing;
                }
                break;
            }
            Partial partial(0);
            partial[dimension] = 1;
            return samplePartial(this->mGrid->getCoordAt(gridCoord), partial);
        }

        /**
         * @brief Numerically estimates the second-order partial derivative using finite differences on a grid for a given dimension. For accuracy=2, the coefficients are hard-coded. For anything above, it falls back to sample_dx, which computes the coefficients automatically.
         * @param gridCoord Grid coordinate at which to estimate the finite difference.
         * @return Estimated partial derivative.
         */
        [[nodiscard]] Value estimate_2nd_partial(const typename GridType::GridCoord& gridCoord, int dimension) const
        {
            double spacing             = this->mGrid->getSpacing()[dimension];
            auto resolution            = this->mGrid->getResolution();
            Eigen::Index linearIndex   = this->mGrid->getLinearIndex(gridCoord);
            auto strides               = this->mGrid->getPointStrides();
            EBoundaryBehavior boundary = this->mBoundaryBehavior[dimension];

            switch (this->mAccuracy)
            {
            case 2:
                if (gridCoord[dimension] > 0 && gridCoord[dimension] < resolution[dimension] - 1 && resolution[dimension] >= 2 && boundary == EBoundaryBehavior::Clamp)
                {
                    return (+1 * mArray->getValue(linearIndex - strides[dimension]).template cast<double>() +
                            -2 * mArray->getValue(linearIndex).template cast<double>() +
                            +1 * mArray->getValue(linearIndex + strides[dimension]).template cast<double>()) /
                           (spacing * spacing);
                }
                else if (gridCoord[dimension] == 0 && resolution[dimension] >= 3 && boundary == EBoundaryBehavior::Clamp)
                {
                    return (+2 * mArray->getValue(linearIndex).template cast<double>() +
                            -5 * mArray->getValue(linearIndex + strides[dimension]).template cast<double>() +
                            +4 * mArray->getValue(linearIndex + 2 * strides[dimension]).template cast<double>() +
                            -1 * mArray->getValue(linearIndex + 3 * strides[dimension]).template cast<double>()) /
                           (spacing * spacing);
                }
                else if (gridCoord[dimension] == resolution[dimension] - 1 && resolution[dimension] >= 3 && boundary == EBoundaryBehavior::Clamp)
                {
                    return (-1 * mArray->getValue(linearIndex - 3 * strides[dimension]).template cast<double>() +
                            +4 * mArray->getValue(linearIndex - 2 * strides[dimension]).template cast<double>() +
                            -5 * mArray->getValue(linearIndex - strides[dimension]).template cast<double>() +
                            +2 * mArray->getValue(linearIndex).template cast<double>()) /
                           (spacing * spacing);
                }
                break;
            }
            Partial partial(0);
            partial[dimension] = 2;
            return samplePartial(this->mGrid->getCoordAt(gridCoord), partial);
        }

        /**
         * @brief Calculates weights for finite difference estimation.
         * The implementation is an adaptation of https://github.com/bjodah/finitediff (BSD license) for equispaced grids. See the paper "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids", Bengt Fornberg, Mathematics of compuation, 51, 184, 1988, 699-706.
         * @tparam Real_t Scalar type for the weights.
         * @param len_g Number of sample points.
         * @param max_deriv Largest derivative to compute.
         * @param sep Grid spacing.
         * @param weights Output weights that are calculated by the function.
         * @param around Location in grid coordinates (zero-based) at which the derivative is evaluated.
         */
        template <typename Real_t>
        void calculateWeights(unsigned len_g, unsigned max_deriv, double sep, Real_t* const weights, Real_t around = 0) const
        {
            // if not enough sample points, return zero.
            if (len_g < max_deriv + 1)
            {
                for (unsigned i = 0; i < len_g * (max_deriv + 1); ++i)
                    weights[i] = 0; // clear weights
                return;
            }
            Real_t c1, c4, c5;
            c1         = 1;
            c4         = (0 - around) * sep;
            weights[0] = 1;
            for (unsigned i = 1; i < len_g * (max_deriv + 1); ++i)
                weights[i] = 0; // clear weights
            for (unsigned i = 1; i < len_g; ++i)
            {
                const int mn = std::min(i, max_deriv);
                Real_t c2    = 1;
                c5           = c4;
                c4           = (i - around) * sep;
                for (unsigned j = 0; j < i; ++j)
                {
                    const Real_t c3   = (i - j) * sep;
                    const Real_t c3_r = 1 / c3;
                    c2                = c2 * c3;
                    if (j == i - 1)
                    {
                        const Real_t c2_r = 1 / c2;
                        for (int k = mn; k >= 1; --k)
                        {
                            const Real_t tmp1      = weights[i - 1 + (k - 1) * len_g];
                            const Real_t tmp2      = weights[i - 1 + k * len_g];
                            weights[i + k * len_g] = c1 * (k * tmp1 - c5 * tmp2) * c2_r;
                        }
                        weights[i] = -c1 * c5 * weights[i - 1] * c2_r;
                    }
                    for (unsigned k = mn; k >= 1; --k)
                    {
                        const Real_t tmp1      = weights[j + k * len_g];
                        const Real_t tmp2      = weights[j + (k - 1) * len_g];
                        weights[j + k * len_g] = (c4 * tmp1 - k * tmp2) * c3_r;
                    }
                    weights[j] = c4 * weights[j] * c3_r;
                }
                c1 = c2;
            }
        }

        /**
         * @brief Gets the data array.
         * @return Underlying array that contains the data.
         */
        [[nodiscard]] inline virtual std::shared_ptr<typename BaseRegularField<TBaseType, TGridType>::ArrayType> getArrayImpl() { return mArray; }

        /**
         * @brief Gets the data array.
         * @return Underlying array that contains the data.
         */
        [[nodiscard]] inline virtual std::shared_ptr<const typename BaseRegularField<TBaseType, TGridType>::ArrayType> getArrayImpl() const { return mArray; }

        /**
         * @brief Data that is stored on the grid points.
         */
        std::shared_ptr<TArrayType> mArray;
    };
}
