#pragma once

namespace vislab
{
    /**
     * @brief Primitive topology.
     */
    enum class EPrimitiveTopology
    {
        /**
         * @brief List of triangles.
         */
        TriangleList,

        /**
         * @brief Strip of triangles.
         */
        TriangleStrip,

        /**
         * @brief List of triangular patches.
         */
        PatchList3,

        /**
         * @brief List of quad patches.
         */
        PatchList4,
    };
}
