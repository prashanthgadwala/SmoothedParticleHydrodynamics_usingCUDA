#pragma once

#include "base_field_fwd.hpp"

#include <vislab/core/types.hpp>

namespace vislab
{
    /**
     * @brief Base class for tensor fields in arbitrary dimension.
     * @tparam SpatialDimensions Number of spatial dimensions.
     * @tparam DomainDimensions Total number of dimensions.
     */
    template <typename TTensor, int64_t SpatialDimensions, int64_t DomainDimensions>
    using ITensorField = BaseField<TTensor, SpatialDimensions, DomainDimensions>;

    /**
     * @brief Base class for 2D tensor fields.
     * @tparam DomainDimensions Total number of dimensions. (2=steady, 3=unsteady)
     */
    template <int64_t TDomainDimensions>
    using ITensorField2d = ITensorField<Eigen::Matrix2d, 2, TDomainDimensions>;

    /**
     * @brief Base class for 3D tensor fields.
     * @tparam DomainDimensions Total number of dimensions. (3=steady, 4=unsteady)
     */
    template <int64_t TDomainDimensions>
    using ITensorField3d = ITensorField<Eigen::Matrix3d, 3, TDomainDimensions>;

    /**
     * @brief Base class for 2D steady tensor fields.
     */
    using ISteadyTensorField2d = ITensorField2d<2>;

    /**
     * @brief Base class for 3D steady tensor fields.
     */
    using ISteadyTensorField3d = ITensorField3d<3>;

    /**
     * @brief Base class for 2D unsteady vector fields.
     */
    using IUnsteadyTensorField2d = ITensorField2d<3>;

    /**
     * @brief Base class for 3D unsteady vector fields.
     */
    using IUnsteadyTensorField3d = ITensorField3d<4>;
}
