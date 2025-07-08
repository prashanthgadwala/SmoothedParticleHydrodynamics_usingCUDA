#pragma once

#include <vislab/core/object.hpp>

namespace vislab
{
    /**
     * @brief Base class that provides unique identifiers for resources that remain the same throughout the lifetime of an object. No RUID is handed out twice, even when the resource holding it previously is deleted.
    */
    class Resource : public Concrete<Resource>
    {
    public:
        /**
         * @brief Constructor, which assigns the uid.
         */
        Resource();

        /**
         * @brief Unique identifier of this resource, which remains the same throughout the lifetime of this resource.
         */
        const std::size_t uid;
    };
}
