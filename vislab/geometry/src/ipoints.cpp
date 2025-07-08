#include <vislab/geometry/ipoints.hpp>

#include <vislab/geometry/attributes.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    IPoints::IPoints()
        : attributes(std::make_shared<Attributes>())
    {
    }

    IPoints::IPoints(const IPoints& other)
        : attributes(other.attributes->clone())
    {
    }

    IPoints::~IPoints() {}

    std::shared_ptr<IArray> IPoints::getVertices()
    {
        return getVerticesImpl();
    }

    std::shared_ptr<const IArray> IPoints::getVertices() const
    {
        return getVerticesImpl();
    }

    void IPoints::clear()
    {
        attributes->clear();
    }

    void IPoints::append(const IPoints* points)
    {
        auto vertices = getVertices();
        vertices->append(points->getVertices().get());
        assert(this->attributes->getSize() == points->attributes->getSize());
        for (std::size_t a = 0; a < this->attributes->getSize(); ++a)
            attributes->getByIndex(a)->append(points->attributes->getByIndex(a).get());
    }

    void IPoints::prepend(const IPoints* points)
    {
        auto vertices = getVertices();
        vertices->prepend(points->getVertices().get());
        assert(this->attributes->getSize() == points->attributes->getSize());
        for (std::size_t a = 0; a < this->attributes->getSize(); ++a)
            attributes->getByIndex(a)->prepend(points->attributes->getByIndex(a).get());
    }

    void IPoints::removeFirst(std::size_t n)
    {
        auto vertices = getVertices();
        vertices->removeFirst(n);
        for (std::size_t a = 0; a < this->attributes->getSize(); ++a)
            attributes->getByIndex(a)->removeFirst(n);
    }

    void IPoints::removeLast(std::size_t n)
    {
        auto vertices = getVertices();
        vertices->removeLast(n);
        for (std::size_t a = 0; a < this->attributes->getSize(); ++a)
            attributes->getByIndex(a)->removeLast(n);
    }

    void IPoints::reverse()
    {
        auto vertices = getVertices();
        vertices->reverse();
        for (std::size_t a = 0; a < this->attributes->getSize(); ++a)
            attributes->getByIndex(a)->reverse();
    }

    bool IPoints::isEqual(const IPoints* other) const
    {
        auto vertices = getVertices();
        if (!vertices->isEqual(other->getVertices().get()))
            return false;
        if (this->attributes->getSize() != other->attributes->getSize())
            return false;
        for (std::size_t i = 0; i < other->attributes->getSize(); ++i)
            if (!attributes->getByIndex(i)->isEqual(other->attributes->getByIndex(i).get()))
                return false;
        return true;
    }

    void IPoints::serialize(IArchive& archive)
    {
        archive("Attributes", attributes);
    }
}
