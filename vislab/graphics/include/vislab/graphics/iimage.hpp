#pragma once

#include <vislab/core/data.hpp>
#include <vislab/core/types.hpp>

namespace vislab
{
    /**
     * @brief Interface for image data structures in any dimension.
     */
    class IImage : public Interface<IImage, Data>
    {
    public:
        /**
         * @brief Gets the total number of pixels.
         * @return Number of pixels.
         */
        [[nodiscard]] virtual Eigen::Index getNumPixels() const = 0;

        /**
         * @brief Gets the number of channels stored in the image.
         * @return Number of color channels.
         */
        [[nodiscard]] virtual Eigen::Index getNumChannels() const = 0;

        /**
         * @brief Gets the resolution of the image.
         * @return Image resolution.
         */
        [[nodiscard]] virtual const Eigen::Vector2i& getResolution() const = 0;

        /**
         * @brief Sets the resolution of the image.
         * @param resolution Image resolution to set.
         */
        virtual void setResolution(const Eigen::Vector2i& resolution) = 0;

        /**
         * @brief Sets the resolution of the image.
         * @param resx New x resolution.
         * @param resy New y resolution.
         */
        virtual void setResolution(int resx, int resy) = 0;

        /**
         * @brief Sets all values to zero.
         */
        virtual void setZero() = 0;
    };
}
