#pragma once

#include <vislab/core/data.hpp>

#include "spectrum.hpp"
#include "resource.hpp"

namespace vislab
{
    class SurfaceInteraction;

    /**
     * @brief Base class for textures.
     */
    class Texture : public Interface<Texture, Data, Resource>
    {
    public:
        /**
         * @brief Checks if this texture is spatially-varying.
         * @return True, if spatially-varying.
         */
        [[nodiscard]] virtual bool isSpatiallyVarying() const;

        /**
         * @brief Evaluates the texture at a certain texture coordinate in [0,1]^2.
         * @param texCoord Texture coordinate.
         * @param data Data value stored on the surface.
         * @return Sampled color.
         */
        [[nodiscard]] virtual Spectrum evaluate(const Eigen::Vector2d& texCoord, double data) const = 0;
    };
}
