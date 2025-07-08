#pragma once

#include "resource_gl.hpp"

namespace vislab
{
    class Light;

    /**
     * @brief Interface for light sources.
     */
    class LightGl : public Interface<LightGl, ResourceGl<Light>>
    {
    public:
        /**
         * @brief Constructor.
         * @param light Light to be wrapped.
         */
        LightGl(std::shared_ptr<const Light> light);
    };
}
