#include <vislab/geometry/isurfaces.hpp>

namespace vislab
{
    std::shared_ptr<ISurface> ISurfaces::getSurface(std::size_t index)
    {
        return getSurfaceImpl(index);
    }

    std::shared_ptr<const ISurface> ISurfaces::getSurface(std::size_t index) const
    {
        return getSurfaceImpl(index);
    }

    void ISurfaces::addSurface(std::shared_ptr<ISurface> line) { addSurfaceImpl(line); }
}
