#pragma once

#include <vislab/core/array_fwd.hpp>

namespace vislab
{
    /**
     * @brief Class that stores 2D image data (static size).
     * @tparam TArrayType Internal array type that stores the pixel values.
     */
    template <typename TArrayType>
    class Image;

    /**
     * @brief Two-dimensional image storing one float value per pixel.
     */
    using Image1f = Image<Array1f>;

    /**
     * @brief Two-dimensional image storing two float values per pixel.
     */
    using Image2f = Image<Array2f>;

    /**
     * @brief Two-dimensional image storing three float values per pixel.
     */
    using Image3f = Image<Array3f>;

    /**
     * @brief Two-dimensional image storing four float values per pixel.
     */
    using Image4f = Image<Array4f>;

    /**
     * @brief Two-dimensional image storing one double value per pixel.
     */
    using Image1d = Image<Array1d>;

    /**
     * @brief Two-dimensional image storing two double values per pixel.
     */
    using Image2d = Image<Array2d>;

    /**
     * @brief Two-dimensional image storing three double values per pixel.
     */
    using Image3d = Image<Array3d>;

    /**
     * @brief Two-dimensional image storing four double values per pixel.
     */
    using Image4d = Image<Array4d>;
}
