#pragma once

#include <vislab/core/array_fwd.hpp>

namespace vislab
{
    /**
     * @brief Point geometry with specific internal type.
     * @tparam TArrayType Internal array type that stores the vertex positions.
     */
    template <typename TArrayType>
    class Points;

    /**
     * @brief One-dimensional point geometry with vertex positions in float precision.
     */
    using Points1f = Points<Array1f>;

    /**
     * @brief Two-dimensional point geometry with vertex positions in float precision.
     */
    using Points2f = Points<Array2f>;

    /**
     * @brief Three-dimensional point geometry with vertex positions in float precision.
     */
    using Points3f = Points<Array3f>;

    /**
     * @brief Four-dimensional point geometry with vertex positions in float precision.
     */
    using Points4f = Points<Array4f>;

    /**
     * @brief One-dimensional point geometry with vertex positions in double precision.
     */
    using Points1d = Points<Array1d>;

    /**
     * @brief Two-dimensional point geometry with vertex positions in double precision.
     */
    using Points2d = Points<Array2d>;

    /**
     * @brief Three-dimensional point geometry with vertex positions in double precision.
     */
    using Points3d = Points<Array3d>;

    /**
     * @brief Four-dimensional point geometry with vertex positions in double precision.
     */
    using Points4d = Points<Array4d>;
}
