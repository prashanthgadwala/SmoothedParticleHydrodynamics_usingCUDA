
#pragma once
#include "init_core.hpp"
#include "init_graphics.hpp"
#include "init_geometry.hpp"
#include "init_field.hpp"
#include "init_opengl.hpp"

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
                init_core();
                init_graphics();
                init_geometry();
                init_field();
                init_opengl();

                initialized = true;
            }
        }
    };
}
