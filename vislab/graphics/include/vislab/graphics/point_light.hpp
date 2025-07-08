#pragma once

#include "light.hpp"

#include "spectrum.hpp"

namespace vislab
{
    /**
     * @brief Point light source.
     */
    class PointLight : public Concrete<PointLight, Light>
    {
    public:
        /**
         * @brief Constructor.
         */
        PointLight();

        /**
         * @brief Constructor.
         * @param intensity Initial intensity.
        */
        PointLight(const Spectrum& intensity);

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Radiant intensity "I", i.e., radiant flux per unit solid angle.
         */
        Spectrum intensity;
    };
}
