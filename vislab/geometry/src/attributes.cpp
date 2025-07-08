#include <vislab/geometry/attributes.hpp>

#include <vislab/core/iarchive.hpp>
#include <vislab/core/iarray.hpp>

namespace vislab
{
    Attributes::Attributes() {}

    Attributes::Attributes(const Attributes& other)
    {
        mAttributes.resize(other.getSize());
        for (std::size_t i = 0; i < mAttributes.size(); ++i)
        {
            mAttributes[i] = other.getByIndex(i)->clone();
        }
    }
    Attributes::~Attributes() {}

    std::size_t Attributes::getSize() const { return mAttributes.size(); }
    void Attributes::setSize(std::size_t size) { mAttributes.resize(size); }

    std::shared_ptr<IArray> Attributes::getByIndex(std::size_t index) { return mAttributes[index]; }
    std::shared_ptr<const IArray> Attributes::getByIndex(std::size_t index) const { return mAttributes[index]; }

    std::shared_ptr<IArray> Attributes::getByName(const std::string& name)
    {
        for (auto& attr : mAttributes)
            if (attr->name == name)
                return attr;
        return nullptr;
    }

    std::shared_ptr<const IArray> Attributes::getByName(const std::string& name) const
    {
        for (auto& attr : mAttributes)
            if (attr->name == name)
                return attr;
        return nullptr;
    }

    void Attributes::setByIndex(std::size_t index, std::shared_ptr<IArray> iarray)
    {
        assert(0 <= index && index < mAttributes.size());
        mAttributes[index] = iarray;
    }

    void Attributes::clear()
    {
        mAttributes.clear();
    }

    void Attributes::append(const Attributes* attributes)
    {
        assert(this->getSize() == attributes->getSize());
        for (std::size_t a = 0; a < this->getSize(); ++a)
            this->getByIndex(a)->append(attributes->getByIndex(a).get());
    }

    void Attributes::prepend(const Attributes* attributes)
    {
        assert(this->getSize() == attributes->getSize());
        for (std::size_t a = 0; a < this->getSize(); ++a)
            this->getByIndex(a)->prepend(attributes->getByIndex(a).get());
    }

    void Attributes::add(std::shared_ptr<IArray> _array)
    {
        mAttributes.push_back(_array);
    }

    void Attributes::removeFirst(std::size_t n)
    {
        for (std::size_t a = 0; a < this->getSize(); ++a)
            this->getByIndex(a)->removeFirst(n);
    }

    void Attributes::removeLast(std::size_t n)
    {
        for (std::size_t a = 0; a < this->getSize(); ++a)
            this->getByIndex(a)->removeLast(n);
    }

    void Attributes::reverse()
    {
        for (std::size_t a = 0; a < this->getSize(); ++a)
            this->getByIndex(a)->reverse();
    }

    bool Attributes::isEqual(const Attributes* other) const
    {
        if (this->getSize() != other->getSize())
            return false;
        for (std::size_t i = 0; i < other->getSize(); ++i)
            if (!this->getByIndex(i)->isEqual(other->getByIndex(i).get()))
                return false;
        return true;
    }

    const std::string& Attributes::getName(const std::size_t& attributeIndex) const
    {
        assert(attributeIndex >= 0 && attributeIndex < mAttributes.size() && "Attribute with index does not exist.");
        return mAttributes[attributeIndex]->name;
    }

    void Attributes::serialize(IArchive& archive)
    {
        archive("Attribute", mAttributes);
    }

    bool Attributes::isValid() const
    {
        for (std::size_t a = 0; a < this->getSize(); ++a)
            if (mAttributes[a] == nullptr)
                return false;
        return true;
    }
}
