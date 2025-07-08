#pragma once

#include "iinteractor.hpp"

namespace vislab
{
    /**
     * @brief Interactor that implements a free camera.
     */
    class FreeInteractor : public Concrete<FreeInteractor, IInteractor>
    {
    public:
        /**
         * @brief Constructor.
         */
        explicit FreeInteractor();

        /**
         * @brief Processes mouse events.
         * @param mouseState Current mouse state.
         */
        void onMouseEvent(const MouseState& mouseState) override;

        /**
         * @brief Processes key events.
         * @param keyState Current key state.
         */
        void onKeyEvent(const KeyState& keyState) override;

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Scaling factor for the movement speed.
         */
        double movementSpeed;

        /**
         * @brief Scaling factor for the rotation speed.
         */
        double rotationSpeed;

    private:
        /**
         * @brief Last x coordinate of the cursor.
         */
        int mLastX;

        /**
         * @brief Last y coordinate of the cursor.
         */
        int mLastY;
    };
}
