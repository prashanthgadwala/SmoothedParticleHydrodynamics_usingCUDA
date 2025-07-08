#pragma once

#include "bsdf.hpp"

namespace vislab
{
    class Texture;

    /**
     * @brief A BSDF for diffuse Lambertian shading.
     */
    class DiffuseBSDF : public Concrete<DiffuseBSDF, BSDF>
    {
    public:
        /**
         * @brief Constructor.
         */
        DiffuseBSDF();

        /**
         * @brief Constructor with initial reflectance.
         */
        DiffuseBSDF(std::shared_ptr<Texture> reflectance);

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Reflectance of the material.
         */
        std::shared_ptr<Texture> reflectance;
    };
}
