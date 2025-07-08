#pragma once

#include <vislab/core/array_fwd.hpp>

namespace vislab
{
    /**
     * @brief Set of line geometries with specific internal type.
     * @tparam TArrayType Internal array type that stores the vertex positions.
     */
    template <typename TArrayType>
    class Lines;

    /**
     * @brief One-dimensional set of line geometries with vertex positions in float precision.
     */
    using Lines1f = Lines<Array1f>;

    /**
     * @brief Two-dimensional set of line geometries with vertex positions in float precision.
     */
    using Lines2f = Lines<Array2f>;

    /**
     * @brief Three-dimensional set of line geometries with vertex positions in float precision.
     */
    using Lines3f = Lines<Array3f>;

    /**
     * @brief Four-dimensional set of line geometries with vertex positions in float precision.
     */
    using Lines4f = Lines<Array4f>;

    /**
     * @brief One-dimensional set of line geometries with vertex positions in double precision.
     */
    using Lines1d = Lines<Array1d>;

    /**
     * @brief Two-dimensional set of line geometries with vertex positions in double precision.
     */
    using Lines2d = Lines<Array2d>;

    /**
     * @brief Three-dimensional set of line geometries with vertex positions in double precision.
     */
    using Lines3d = Lines<Array3d>;

    /**
     * @brief Four-dimensional set of line geometries with vertex positions in double precision.
     */
    using Lines4d = Lines<Array4d>;
}
