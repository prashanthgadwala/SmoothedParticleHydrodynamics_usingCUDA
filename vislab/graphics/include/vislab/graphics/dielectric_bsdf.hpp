#pragma once

#include "bsdf.hpp"

#include <Eigen/Eigen>

namespace vislab
{
    class Texture;

    /**
     * @brief A BSDF for a smooth dielectric material.
     */
    class DielectricBSDF : public Concrete<DielectricBSDF, BSDF>
    {
    public:
        /**
         * @brief Constructor.
         */
        DielectricBSDF();

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Interior index of reflection.
         */
        double interior_ior;

        /**
         * @brief Exterios index of reflection.
         */
        double exterior_ior;

        /**
         * @brief Relative index of refraction from the exterior to the interior.
         */
        double eta;

        /**
         * @brief Specular reflectance of the material.
         */
        std::shared_ptr<Texture> specularReflectance;

        /**
         * @brief Specular transmittance of the material.
         */
        std::shared_ptr<Texture> specularTransmittance;
    };
}
