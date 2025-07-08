#pragma once

#include "component.hpp"

namespace vislab
{
    /**
     * @brief Base class for bidirectional scattering distribution functions.
     */
    class BSDF : public Interface<BSDF, Component>
    {
    };
}
