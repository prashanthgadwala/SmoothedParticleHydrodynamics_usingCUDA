#pragma once

#include "light.hpp"

#include "spectrum.hpp"

namespace vislab
{
    /**
     * @brief Area light source.
     */
    class AreaLight : public Concrete<AreaLight, Light>
    {
    public:
        /**
         * @brief Constructor.
         */
        AreaLight();

        /**
         * @brief Constructor.
         * @param radiance Initial radiance.
        */
        AreaLight(const Spectrum& radiance);

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Emitted radiance.
         */
        Spectrum radiance;
    };
}
