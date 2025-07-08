#pragma once

#include <vislab/core/array_fwd.hpp>

namespace vislab
{
    /**
     * @brief Set of surface geometries with specific internal type.
     * @tparam TVertexArrayType Internal array type that stores the vertex buffer positions.
     * @tparam TIndexArrayType Internal array type that stores the index buffer values.
     */
    template <typename TVertexArrayType, typename TIndexArrayType>
    class Surfaces;

    /**
     * @brief One-dimensional sets of surface geometries with vertex positions in float precision.
     */
    using Surfaces1f = Surfaces<Array1f, Array1u>;

    /**
     * @brief Two-dimensional sets of surface geometries with vertex positions in float precision.
     */
    using Surfaces2f = Surfaces<Array2f, Array1u>;

    /**
     * @brief Three-dimensional sets of surface geometries with vertex positions in float precision.
     */
    using Surfaces3f = Surfaces<Array3f, Array1u>;

    /**
     * @brief Four-dimensional sets of surface geometries with vertex positions in float precision.
     */
    using Surfaces4f = Surfaces<Array4f, Array1u>;

    /**
     * @brief One-dimensional sets of surface geometries with vertex positions in double precision.
     */
    using Surfaces1d = Surfaces<Array1d, Array1u>;

    /**
     * @brief Two-dimensional sets of surface geometries with vertex positions in double precision.
     */
    using Surfaces2d = Surfaces<Array2d, Array1u>;

    /**
     * @brief Three-dimensional sets of surface geometries with vertex positions in double precision.
     */
    using Surfaces3d = Surfaces<Array3d, Array1u>;

    /**
     * @brief Four-dimensional sets of surface geometries with vertex positions in double precision.
     */
    using Surfaces4d = Surfaces<Array4d, Array1u>;
}
