#include <vislab/geometry/iline.hpp>

#include <vislab/geometry/attributes.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    ILine::ILine()
        : attributes(std::make_shared<Attributes>())
    {
    }

    ILine::ILine(const ILine& other)
        : attributes(other.attributes->clone())
    {
    }

    ILine::~ILine() {}

    std::shared_ptr<IArray> ILine::getVertices()
    {
        return getVerticesImpl();
    }

    std::shared_ptr<const IArray> ILine::getVertices() const
    {
        return getVerticesImpl();
    }

    void ILine::clear()
    {
        attributes->clear();
    }

    void ILine::append(const ILine* line)
    {
        auto vertices = getVertices();
        vertices->append(line->getVertices().get());
        assert(this->attributes->getSize() == line->attributes->getSize());
        for (std::size_t a = 0; a < this->attributes->getSize(); ++a)
            attributes->getByIndex(a)->append(line->attributes->getByIndex(a).get());
    }

    void ILine::prepend(const ILine* line)
    {
        auto vertices = getVertices();
        vertices->prepend(line->getVertices().get());
        assert(this->attributes->getSize() == line->attributes->getSize());
        for (std::size_t a = 0; a < this->attributes->getSize(); ++a)
            attributes->getByIndex(a)->prepend(line->attributes->getByIndex(a).get());
    }

    void ILine::removeFirst(std::size_t n)
    {
        auto vertices = getVertices();
        vertices->removeFirst(n);
        for (std::size_t a = 0; a < this->attributes->getSize(); ++a)
            attributes->getByIndex(a)->removeFirst(n);
    }

    void ILine::removeLast(std::size_t n)
    {
        auto vertices = getVertices();
        vertices->removeLast(n);
        for (std::size_t a = 0; a < this->attributes->getSize(); ++a)
            attributes->getByIndex(a)->removeLast(n);
    }

    void ILine::reverse()
    {
        auto vertices = getVertices();
        vertices->reverse();
        for (std::size_t a = 0; a < this->attributes->getSize(); ++a)
            attributes->getByIndex(a)->reverse();
    }

    bool ILine::isEqual(const ILine* other) const
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

    void ILine::serialize(IArchive& archive)
    {
        archive("Attributes", attributes);
    }
}
