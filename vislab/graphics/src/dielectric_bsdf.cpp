#include <vislab/graphics/dielectric_bsdf.hpp>

#include <vislab/graphics/const_texture.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    DielectricBSDF::DielectricBSDF()
        : interior_ior(1.5046)
        , exterior_ior(1.000277)
        , eta(1.5046 / 1.000277)
        , specularReflectance(std::make_shared<ConstTexture>(Spectrum(1.0, 1.0, 1.0)))
        , specularTransmittance(std::make_shared<ConstTexture>(Spectrum(1.0, 1.0, 1.0)))
    {
    }

    void DielectricBSDF::serialize(IArchive& archive)
    {
        archive("interior_ior", interior_ior);
        archive("exterior_ior", exterior_ior);
        archive("eta", eta);
        archive("specularReflectance", specularReflectance);
        archive("specularTransmittance", specularTransmittance);
    }
}
