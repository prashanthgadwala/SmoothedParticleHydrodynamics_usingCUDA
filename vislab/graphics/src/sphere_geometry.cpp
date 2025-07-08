#include <vislab/graphics/sphere_geometry.hpp>

#include <vislab/core/array.hpp>
#include <vislab/core/iarchive.hpp>

namespace vislab
{
    SphereGeometry::SphereGeometry(float radiusScale)
        : radiusScale(radiusScale)
        , radius(nullptr)
        , data(nullptr)
    {
        // by default, there is one point at (0,0,0)
        positions = std::make_shared<Array3f>();
        positions->append(Eigen::Vector3f(0, 0, 0));
    }

    Eigen::AlignedBox3d SphereGeometry::objectBounds() const
    {
        // initially create an empty box
        Eigen::AlignedBox3d result;
        result.setEmpty();

        // get data and iterate all points
        Eigen::Index numVertices = positions->getSize();
        for (Eigen::Index iv = 0; iv < numVertices; ++iv)
        {
            // get radius
            double r = 1;
            if (radius && iv < radius->getSize())
                r = radius->getValue(iv).x();
            r *= radiusScale;
            // get center
            Eigen::Vector3f center = positions->getValue(iv);
            // expand the box
            result.extend(center.cast<double>() - Eigen::Vector3d(r, r, r));
            result.extend(center.cast<double>() + Eigen::Vector3d(r, r, r));
        }
        return result;
    }

    void SphereGeometry::serialize(IArchive& archive)
    {
        archive("Positions", positions);
        archive("RadiusScale", radiusScale);
        archive("Radius", radius);
        archive("Data", data);
    }
}
