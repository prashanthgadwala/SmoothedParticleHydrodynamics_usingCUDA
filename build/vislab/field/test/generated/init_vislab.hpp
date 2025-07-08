
#pragma once
#include "init_field.hpp"
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
                init_field();
                init_core();

                initialized = true;
            }
        }
    };
}
