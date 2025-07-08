#pragma once

#include <stdint.h>

namespace vislab
{
    /**
     * @brief Abstract base class for two-dimensional images with custom number of color components per pixel.
     * @tparam TChannels Number of color components per pixel.
     */
    template <int64_t TChannels>
    class BaseImage;

    /**
     * @brief Abstract two-dimensional base image with one component per pixel.
     */
    using IImage1 = BaseImage<1>;

    /**
     * @brief Abstract two-dimensional base image with two components per pixel.
     */
    using IImage2 = BaseImage<2>;

    /**
     * @brief Abstract two-dimensional base image with three components per pixel.
     */
    using IImage3 = BaseImage<3>;

    /**
     * @brief Abstract two-dimensional base image with four components per pixel.
     */
    using IImage4 = BaseImage<4>;
}
