#pragma once

#include "texture.hpp"

#include <vislab/core/event.hpp>
#include <vislab/core/transfer_function.hpp>
#include <vislab/field/regular_field_fwd.hpp>

namespace vislab
{
    /**
     * @brief Texture that is derived via a transfer function from a scalar field.
     */
    class ColormapTexture : public Concrete<ColormapTexture, Texture>
    {
    public:
        /**
         * @brief Constructor a default color map from [0,1] -> black-white
         */
        ColormapTexture();

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
         * @brief Transfer function that maps the scalar value to color.
         */
        TransferFunction4d transferFunction;

        /**
         * @brief Scalar field that is mapped to color.
         */
        std::shared_ptr<RegularSteadyScalarField2f> scalarField;
    };
}
