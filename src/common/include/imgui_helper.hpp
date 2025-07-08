#pragma once

#include <imgui.h>

namespace physsim
{
    /**
     * @brief Collection of helper functions for the creation of UI elements.
     */
    class ImguiHelper
    {
    public:
        /**
         * @brief Creates a toggle button.
         * @param str_id unique ID internally used to keep track of the state.
         * @param v Pointer to bool variable that holds the state.
         */
        static void toggleButton(const char* str_id, bool* v);
    };
}
