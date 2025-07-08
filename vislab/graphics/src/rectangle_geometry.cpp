#include <vislab/graphics/rectangle_geometry.hpp>

namespace vislab
{
    Eigen::AlignedBox3d RectangleGeometry::objectBounds() const
    {
        return Eigen::AlignedBox3d(
            Eigen::Vector3d(-1, -1, 0),
            Eigen::Vector3d(1, 1, 0));
    }

    void RectangleGeometry::serialize(IArchive& archive)
    {
    }
}
