#pragma once

#include <vislab/core/array_fwd.hpp>

namespace vislab
{
    /**
     * @brief Line geometry with specific internal type.
     * @tparam TArrayType Internal array type that stores the vertex positions.
     */
    template <typename TArrayType>
    class Line;

    /**
     * @brief One-dimensional line geometry with vertex positions in float precision.
     */
    using Line1f = Line<Array1f>;

    /**
     * @brief Two-dimensional line geometry with vertex positions in float precision.
     */
    using Line2f = Line<Array2f>;

    /**
     * @brief Three-dimensional line geometry with vertex positions in float precision.
     */
    using Line3f = Line<Array3f>;

    /**
     * @brief Four-dimensional line geometry with vertex positions in float precision.
     */
    using Line4f = Line<Array4f>;

    /**
     * @brief One-dimensional line geometry with vertex positions in double precision.
     */
    using Line1d = Line<Array1d>;

    /**
     * @brief Two-dimensional line geometry with vertex positions in double precision.
     */
    using Line2d = Line<Array2d>;

    /**
     * @brief Three-dimensional line geometry with vertex positions in double precision.
     */
    using Line3d = Line<Array3d>;

    /**
     * @brief Four-dimensional line geometry with vertex positions in double precision.
     */
    using Line4d = Line<Array4d>;
}
