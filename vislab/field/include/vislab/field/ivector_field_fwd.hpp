#pragma once

#include "base_field_fwd.hpp"

#include <vislab/core/types.hpp>

namespace vislab
{
    /**
     * @brief Base class for vector fields in arbitrary dimension.
     * @tparam SpatialDimensions Number of spatial dimensions.
     * @tparam DomainDimensions Total number of dimensions.
     */
    template <typename TVector, int64_t SpatialDimensions, int64_t DomainDimensions>
    using IVectorField = BaseField<TVector, SpatialDimensions, DomainDimensions>;

    /**
     * @brief Base class for 2D vector fields.
     * @tparam DomainDimensions Total number of dimensions. (2=steady, 3=unsteady)
     */
    template <int64_t TDomainDimensions>
    using IVectorField2d = IVectorField<Eigen::Vector2d, 2, TDomainDimensions>;

    /**
     * @brief Base class for 3D vector fields.
     * @tparam DomainDimensions Total number of dimensions. (3=steady, 4=unsteady)
     */
    template <int64_t TDomainDimensions>
    using IVectorField3d = IVectorField<Eigen::Vector3d, 3, TDomainDimensions>;

    /**
     * @brief Base class for 2D steady vector fields.
     */
    using ISteadyVectorField2d = IVectorField2d<2>;

    /**
     * @brief Base class for 3D steady vector fields.
     */
    using ISteadyVectorField3d = IVectorField3d<3>;

    /**
     * @brief Base class for 2D unsteady vector fields.
     */
    using IUnsteadyVectorField2d = IVectorField2d<3>;

    /**
     * @brief Base class for 3D unsteady vector fields.
     */
    using IUnsteadyVectorField3d = IVectorField3d<4>;
}
