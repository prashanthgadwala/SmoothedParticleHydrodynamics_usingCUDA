#include <vislab/core/iarray.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    IArray::IArray()
        : name("")
    {
    }

    IArray::IArray(const std::string& _name)
        : name(_name)
    {
    }

    void IArray::serialize(IArchive& archive)
    {
        archive("Name", name);
    }
}
