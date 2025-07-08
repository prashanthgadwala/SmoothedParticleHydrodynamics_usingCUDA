#pragma once

#include "base_regular_field_fwd.hpp"

#include <vislab/core/array_fwd.hpp>

namespace vislab
{
    /**
     * @brief Base class for fields defined on regular grids.
     * @tparam TBaseType Field type to discretize onto a regular grid.
     * @tparam TGridType Grid type to discretize on.
     * @tparam TArrayType Internal array type that stores values.
     */
    template <class TBaseType, typename TGridType, typename TArrayType>
    class RegularField;

    /**
     * @brief Two-dimensional steady scalar field on a regular grid.
     */
    using RegularSteadyScalarField2d = RegularField<ISteadyScalarField2d, RegularGrid2d, Array1d>;

    /**
     * @brief Two-dimensional steady scalar field on a regular grid.
     */
    using RegularSteadyScalarField2f = RegularField<ISteadyScalarField2d, RegularGrid2d, Array1f>;

    /**
     * @brief Three-dimensional steady scalar field on a regular grid.
     */
    using RegularSteadyScalarField3d = RegularField<ISteadyScalarField3d, RegularGrid3d, Array1d>;

    /**
     * @brief Three-dimensional steady scalar field on a regular grid.
     */
    using RegularSteadyScalarField3f = RegularField<ISteadyScalarField3d, RegularGrid3d, Array1f>;

    /**
     * @brief Two-dimensional unsteady scalar field on a regular grid.
     */
    using RegularUnsteadyScalarField2d = RegularField<IUnsteadyScalarField2d, RegularGrid3d, Array1d>;

    /**
     * @brief Two-dimensional unsteady scalar field on a regular grid.
     */
    using RegularUnsteadyScalarField2f = RegularField<IUnsteadyScalarField2d, RegularGrid3d, Array1f>;

    /**
     * @brief Three-dimensional unsteady scalar field on a regular grid.
     */
    using RegularUnsteadyScalarField3d = RegularField<IUnsteadyScalarField3d, RegularGrid4d, Array1d>;

    /**
     * @brief Three-dimensional unsteady scalar field on a regular grid.
     */
    using RegularUnsteadyScalarField3f = RegularField<IUnsteadyScalarField3d, RegularGrid4d, Array1f>;

    /**
     * @brief Two-dimensional steady vector field on a regular grid.
     */
    using RegularSteadyVectorField2d = RegularField<ISteadyVectorField2d, RegularGrid2d, Array2d>;

    /**
     * @brief Two-dimensional steady vector field on a regular grid.
     */
    using RegularSteadyVectorField2f = RegularField<ISteadyVectorField2d, RegularGrid2d, Array2f>;

    /**
     * @brief Three-dimensional steady vector field on a regular grid.
     */
    using RegularSteadyVectorField3d = RegularField<ISteadyVectorField3d, RegularGrid3d, Array3d>;

    /**
     * @brief Three-dimensional steady vector field on a regular grid.
     */
    using RegularSteadyVectorField3f = RegularField<ISteadyVectorField3d, RegularGrid3d, Array3f>;

    /**
     * @brief Two-dimensional unsteady vector field on a regular grid.
     */
    using RegularUnsteadyVectorField2d = RegularField<IUnsteadyVectorField2d, RegularGrid3d, Array2d>;

    /**
     * @brief Two-dimensional unsteady vector field on a regular grid.
     */
    using RegularUnsteadyVectorField2f = RegularField<IUnsteadyVectorField2d, RegularGrid3d, Array2f>;

    /**
     * @brief Three-dimensional unsteady vector field on a regular grid.
     */
    using RegularUnsteadyVectorField3d = RegularField<IUnsteadyVectorField3d, RegularGrid4d, Array3d>;

    /**
     * @brief Three-dimensional unsteady vector field on a regular grid.
     */
    using RegularUnsteadyVectorField3f = RegularField<IUnsteadyVectorField3d, RegularGrid4d, Array3f>;

    /**
     * @brief Two-dimensional steady tensor field on a regular grid.
     */
    using RegularSteadyTensorField2d = RegularField<ISteadyTensorField2d, RegularGrid2d, Array2x2d>;

    /**
     * @brief Two-dimensional steady tensor field on a regular grid.
     */
    using RegularSteadyTensorField2f = RegularField<ISteadyTensorField2d, RegularGrid2d, Array2x2f>;

    /**
     * @brief Three-dimensional steady tensor field on a regular grid.
     */
    using RegularSteadyTensorField3d = RegularField<ISteadyTensorField3d, RegularGrid3d, Array3x3d>;

    /**
     * @brief Three-dimensional steady tensor field on a regular grid.
     */
    using RegularSteadyTensorField3f = RegularField<ISteadyTensorField3d, RegularGrid3d, Array3x3f>;

    /**
     * @brief Two-dimensional unsteady tensor field on a regular grid.
     */
    using RegularUnsteadyTensorField2d = RegularField<IUnsteadyTensorField2d, RegularGrid3d, Array2x2d>;

    /**
     * @brief Two-dimensional unsteady tensor field on a regular grid.
     */
    using RegularUnsteadyTensorField2f = RegularField<IUnsteadyTensorField2d, RegularGrid3d, Array2x2f>;

    /**
     * @brief Three-dimensional unsteady tensor field on a regular grid.
     */
    using RegularUnsteadyTensorField3d = RegularField<IUnsteadyTensorField3d, RegularGrid4d, Array3x3d>;

    /**
     * @brief Three-dimensional unsteady tensor field on a regular grid.
     */
    using RegularUnsteadyTensorField3f = RegularField<IUnsteadyTensorField3d, RegularGrid4d, Array3x3f>;
}
