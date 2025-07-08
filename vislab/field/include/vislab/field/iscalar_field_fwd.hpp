#pragma once

#include "base_field_fwd.hpp"

#include <vislab/core/types.hpp>

namespace vislab
{
    /**
     * @brief Base class for scalar fields in arbitrary dimension.
     * @tparam SpatialDimensions Number of spatial dimensions.
     * @tparam DomainDimensions Total number of dimensions.
     */
    template <int64_t SpatialDimensions, int64_t DomainDimensions>
    using IScalarField = BaseField<Eigen::Vector1d, SpatialDimensions, DomainDimensions>;

    /**
     * @brief Base class for 2D scalar fields.
     * @tparam TDomainDimensions Total number of dimensions. (2=steady, 3=unsteady)
     */
    template <int64_t TDomainDimensions>
    using IScalarField2d = IScalarField<2, TDomainDimensions>;

    /**
     * @brief Base class for 3D scalar fields.
     * @tparam TDomainDimensions Total number of dimensions. (3=steady, 4=unsteady)
     */
    template <int64_t TDomainDimensions>
    using IScalarField3d = IScalarField<3, TDomainDimensions>;

    /**
     * @brief Base class for 2D steady scalar fields.
     */
    using ISteadyScalarField2d = IScalarField2d<2>;

    /**
     * @brief Base class for 3D steady scalar fields.
     */
    using ISteadyScalarField3d = IScalarField3d<3>;

    /**
     * @brief Base class for 2D unsteady scalar fields.
     */
    using IUnsteadyScalarField2d = IScalarField2d<3>;

    /**
     * @brief Base class for 3D unsteady scalar fields.
     */
    using IUnsteadyScalarField3d = IScalarField3d<4>;
}
