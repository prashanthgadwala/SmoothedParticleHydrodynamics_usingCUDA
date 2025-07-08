#include <vislab/graphics/tags.hpp>

#include <vislab/graphics/tag.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    void Tags::serialize(IArchive& archive)
    {
        archive("Tags", mTags);
    }
}
