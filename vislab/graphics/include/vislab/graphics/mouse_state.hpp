#pragma once

namespace vislab
{
    /**
     * @brief Represents the current mouse state.
     */
    struct MouseState
    {
        bool leftDown;
        bool leftIsDown;
        bool leftUp;
        bool rightDown;
        bool rightIsDown;
        bool rightUp;
        bool middleDown;
        bool middleIsDown;
        bool middleUp;
        bool shiftDown;
        bool ctrlDown;
        int x;
        int y;
        int width;
        int height;
        double scrollDelta;
    };
}
