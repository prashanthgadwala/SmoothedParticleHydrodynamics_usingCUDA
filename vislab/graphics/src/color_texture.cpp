#include <vislab/graphics/colormap_texture.hpp>

#include <vislab/core/iarchive.hpp>
#include <vislab/field/regular_field.hpp>

namespace vislab
{
    ColormapTexture::ColormapTexture()
    {
        transferFunction.minValue = 0;
        transferFunction.maxValue = 1;
        transferFunction.values.insert(std::make_pair(0, Eigen::Vector4d::Zero()));
        transferFunction.values.insert(std::make_pair(1, Eigen::Vector4d::Ones()));
    }

    void ColormapTexture::serialize(IArchive& archive)
    {
        archive("TransferFunction", transferFunction);
        archive("ScalarField", scalarField);
    }

    Spectrum ColormapTexture::evaluate(const Eigen::Vector2d& texCoord, double data) const
    {
        if (scalarField)
        {
            Eigen::AlignedBox2d domain = scalarField->getDomain();
            Eigen::Vector2d pos        = domain.min() + (domain.max() - domain.min()).cwiseProduct(texCoord);
            data                       = scalarField->sample(pos).x();
        }
        Eigen::Vector4d color = transferFunction.map(data);
        return color.xyz();
    }
}
