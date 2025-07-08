#include <vislab/graphics/area_light.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    AreaLight::AreaLight()
        : radiance(Spectrum::Zero())
    {
    }

    AreaLight::AreaLight(const Spectrum& radiance)
        : radiance(radiance)
    {
    }

    void AreaLight::serialize(IArchive& archive)
    {
        archive("Radiance", radiance);
    }
}
