#pragma once

#include "base_image_fwd.hpp"

#include "iimage.hpp"

namespace vislab
{
    /**
     * @brief Abstract base class for two-dimensional images with custom number of color components per pixel.
     * @tparam TChannels Number of color components per pixel.
     * @tparam TD Dummy parameter needed to make this type reflection compatible.
     */
    template <int64_t TChannels>
    class BaseImage : public Interface<BaseImage<TChannels>, IImage>
    {
    public:
        /**
         * @brief Gets the number of color channels.
         */
        static const int64_t Channels = TChannels;

        /**
         * @brief Gets the number of channels stored in the image.
         * @return Number of color channels.
         */
        [[nodiscard]] inline Eigen::Index getNumChannels() const override { return Channels; }
    };
}
