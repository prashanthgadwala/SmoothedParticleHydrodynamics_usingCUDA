#include <vislab/graphics/actor.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    Actor::Actor()
    //    : reverseOrientation(false)
    {
    }

    Actor::Actor(const std::string& name)
        : name(name)
    {
    }

    void Actor::serialize(IArchive& archive)
    {
        archive("Name", name);
        archive("Components", components);
        archive("Tags", tags);
        //   archive("ReverseOrientation", reverseOrientation);
    }
}
