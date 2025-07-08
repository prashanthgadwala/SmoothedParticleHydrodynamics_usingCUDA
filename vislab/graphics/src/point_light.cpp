#include <vislab/graphics/point_light.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    PointLight::PointLight()
        : intensity(Spectrum::Zero())
    {
    }

    PointLight::PointLight(const Spectrum& intensity)
        : intensity(intensity)
    {
    }

    void PointLight::serialize(IArchive& archive)
    {
        archive("Intensity", intensity);
    }
}
