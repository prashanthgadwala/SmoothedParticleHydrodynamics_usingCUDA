#include <vislab/graphics/const_texture.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    ConstTexture::ConstTexture()
        : color(Spectrum(1, 1, 1))
    {
    }

    ConstTexture::ConstTexture(const Spectrum& spectrum)
        : color(spectrum)
    {
    }

    void ConstTexture::serialize(IArchive& archive)
    {
        archive("Color", color);
    }

    Spectrum ConstTexture::evaluate(const Eigen::Vector2d& texCoord, double data) const
    {
        return color;
    }
}
