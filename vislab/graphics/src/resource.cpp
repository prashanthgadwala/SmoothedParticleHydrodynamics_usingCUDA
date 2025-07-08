#include <vislab/graphics/resource.hpp>

namespace vislab
{
    static std::size_t generateUniqueIdentifier()
    {
        static std::size_t global_uid = 0;
        return global_uid++;
    }

    Resource::Resource()
        : uid(generateUniqueIdentifier())
    {
    }
}
