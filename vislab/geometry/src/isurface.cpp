#include <vislab/geometry/isurface.hpp>

#include <vislab/geometry/attributes.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    ISurface::ISurface()
        : attributes(std::make_shared<Attributes>())
    {
    }

    ISurface::ISurface(const ISurface& other)
        : attributes(other.attributes->clone())
    {
    }

    ISurface::~ISurface() {}

    std::shared_ptr<IArray> ISurface::getPositions()
    {
        return getPositionsImpl();
    }

    std::shared_ptr<const IArray> ISurface::getPositions() const
    {
        return getPositionsImpl();
    }

    void ISurface::clear()
    {
        attributes->clear();
    }

    bool ISurface::isEqual(const ISurface* other) const
    {
        if (this->attributes->getSize() != other->attributes->getSize())
            return false;

        for (std::size_t i = 0; i < other->attributes->getSize(); ++i)
            if (!attributes->getByIndex(i)->isEqual(other->attributes->getByIndex(i).get()))
                return false;
        return true;
    }

    void ISurface::serialize(IArchive& archive)
    {
        archive("Attributes", attributes);
    }
}
