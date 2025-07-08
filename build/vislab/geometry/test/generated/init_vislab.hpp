
#pragma once
#include "init_geometry.hpp"
#include "init_core.hpp"

namespace vislab
{
    class Init
    {
    public:
        Init()
        {
            static bool initialized = false;
            if (!initialized)
            {
                init_geometry();
                init_core();

                initialized = true;
            }
        }
    };
}
