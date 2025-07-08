#include <vislab/graphics/diffuse_bsdf.hpp>

#include <vislab/graphics/const_texture.hpp>
#include <vislab/graphics/texture.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    DiffuseBSDF::DiffuseBSDF()
        : reflectance(std::make_shared<ConstTexture>(Spectrum(0.5, 0.5, 0.5)))
    {
    }

    DiffuseBSDF::DiffuseBSDF(std::shared_ptr<Texture> reflectance)
        : reflectance(reflectance)
    {
    }

    void DiffuseBSDF::serialize(IArchive& archive)
    {
        archive("Reflectance", reflectance);
    }
}
