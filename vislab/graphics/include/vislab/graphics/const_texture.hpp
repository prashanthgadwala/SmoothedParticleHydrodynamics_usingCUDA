#pragma once

#include "texture.hpp"

#include "spectrum.hpp"

namespace vislab
{
    /**
     * @brief Spatially-constant texture with a fixed color.
     */
    class ConstTexture : public Concrete<ConstTexture, Texture>
    {
    public:
        /**
         * @brief Constructor with a white color.
         */
        ConstTexture();

        /**
         * @brief Constructor with an initial color.
         * @param spectrum Initial color.
         */
        ConstTexture(const Spectrum& spectrum);

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Evaluates the texture at a certain texture coordinate in [0,1]^2.
         * @param texCoord Texture coordinate.
         * @param data Data value stored on the surface.
         * @return Sampled color.
         */
        [[nodiscard]] Spectrum evaluate(const Eigen::Vector2d& texCoord, double data) const override;

        /**
         * @brief Constant color of this texture.
         */
        Spectrum color;
    };
}
