#include <vislab/graphics/components.hpp>

#include <vislab/graphics/component.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    void Components::serialize(IArchive& archive)
    {
        archive("Components", mComponents);
    }
}
