#pragma once

#include <vislab/core/array_fwd.hpp>

namespace vislab
{
    /**
     * @brief Surface geometry with specific internal type.
     * @tparam TVertexArrayType Internal array type that stores the vertex buffer positions.
     * @tparam TIndexArrayType Internal array type that stores the index buffer values.
     */
    template <typename TVertexArrayType, typename TIndexArrayType>
    class Surface;

    /**
     * @brief One-dimensional surface geometry with vertex positions in float precision.
     */
    using Surface1f = Surface<Array1f, Array1u>;

    /**
     * @brief Two-dimensional surface geometry with vertex positions in float precision.
     */
    using Surface2f = Surface<Array2f, Array1u>;

    /**
     * @brief Three-dimensional surface geometry with vertex positions in float precision.
     */
    using Surface3f = Surface<Array3f, Array1u>;

    /**
     * @brief Four-dimensional surface geometry with vertex positions in float precision.
     */
    using Surface4f = Surface<Array4f, Array1u>;

    /**
     * @brief One-dimensional surface geometry with vertex positions in double precision.
     */
    using Surface1d = Surface<Array1d, Array1u>;

    /**
     * @brief Two-dimensional surface geometry with vertex positions in double precision.
     */
    using Surface2d = Surface<Array2d, Array1u>;

    /**
     * @brief Three-dimensional surface geometry with vertex positions in double precision.
     */
    using Surface3d = Surface<Array3d, Array1u>;

    /**
     * @brief Four-dimensional surface geometry with vertex positions in double precision.
     */
    using Surface4d = Surface<Array4d, Array1u>;
}
