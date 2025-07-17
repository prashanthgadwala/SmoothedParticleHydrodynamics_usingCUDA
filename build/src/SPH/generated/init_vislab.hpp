
#pragma once

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

                initialized = true;
            }
        }
    };
}
