#pragma once

#include "base_field_fwd.hpp"

#include "base_partial.hpp"
#include "ifield.hpp"

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    /**
     * @brief Base class for fields with a specific storage format and certain input dimensions.
     * @tparam TValueType Type that the field is mapping to.
     * @tparam TSpatialDimensions Number of spatial dimensions.
     * @tparam TDomainDimensions Total number of dimensions.
     */
    template <typename TValueType, int64_t TSpatialDimensions, int64_t TDomainDimensions>
    class BaseField : public Interface<BaseField<TValueType, TSpatialDimensions, TDomainDimensions>, IField>
    {
    public:
        /**
         * @brief Type of values in the field.
         */
        using Value = TValueType;

        /**
         * @brief Number of spatial input dimensions.
         */
        static constexpr int64_t SpatialDimensions = TSpatialDimensions;

        /**
         * @brief Flag that determines whether the field is steady.
         */
        static constexpr bool IsSteady = TSpatialDimensions == TDomainDimensions;

        /**
         * @brief Number of input dimensions.
         */
        static constexpr int64_t Dimensions = TDomainDimensions;

        /**
         * @brief Number of output dimensions.
         */
        static constexpr int Components = TValueType::RowsAtCompileTime * TValueType::ColsAtCompileTime;

        /**
         * @brief Number of rows.
         */
        static constexpr int RowsAtCompileTime = TValueType::RowsAtCompileTime;

        /**
         * @brief Number of columns.
         */
        static constexpr int ColsAtCompileTime = TValueType::ColsAtCompileTime;

        /**
         * @brief Type of the bounding box.
         */
        using BoundingBox = Eigen::AlignedBox<double, Dimensions>;

        /**
         * @brief Type of the domain coordinates.
         */
        using DomainCoord = Eigen::Matrix<double, Dimensions, 1>;

        /**
         * @brief Type to specify a partial derivative. This type is used in the samplePartial function to specify the desired derivative of each dimension.
         */
        using Partial = BasePartial<TSpatialDimensions, TDomainDimensions>;

        /**
         * @brief Constructor.
         * @param domain Bounding box of the domain.
         */
        BaseField(const BoundingBox& domain)
            : mDomain(domain)
        {
        }

        /**
         * @brief Gets the number of input dimensions to this field (space + time).
         * @return Total number of dimensions.
         */
        [[nodiscard]] inline Eigen::Index getDimensions() const override { return Dimensions; }

        /**
         * @brief Gets the number of input dimensions to this field (space only).
         * @return Number of spatial dimensions.
         */
        [[nodiscard]] inline Eigen::Index getSpatialDimensions() const override { return SpatialDimensions; }

        /**
         * @brief Samples the field.
         * @param coord Domain location to sample the field at.
         * @return Value at the domain location.
         */
        [[nodiscard]] virtual TValueType sample(const DomainCoord& coord) const = 0;

        /**
         * @brief Samples a partial derivative.
         * @param coord Domain location to sample the field derivative at.
         * @param partial Specifies the desired partial derivative of each dimension.
         * @return Field derivative at domain location.
         */
        [[nodiscard]] virtual TValueType samplePartial(const DomainCoord& coord, const Partial& partial) const = 0;

        /**
         * @brief Gets the domain of the field.
         * @return Bounding box of the domain.
         */
        [[nodiscard]] virtual const BoundingBox& getDomain() const { return mDomain; }

        /**
         * @brief Sets the domain of the field.
         * @param domain New bounding box of the domain.
         */
        virtual void setDomain(const BoundingBox& domain) { mDomain = domain; }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override
        {
            archive("Domain", mDomain);
        }

    protected:
        /**
         * @brief Bounding box of the domain.
         */
        BoundingBox mDomain;
    };
}
