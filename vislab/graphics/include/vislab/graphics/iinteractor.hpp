#pragma once

#include <vislab/core/data.hpp>
#include <vislab/core/event.hpp>

#include <memory>

namespace vislab
{
    struct MouseState;
    struct KeyState;
    class Camera;

    /**
     * @brief Interface for camera interactors.
     */
    class IInteractor : public Interface<IInteractor, Data>
    {
    public:
        /**
         * @brief Constructor.
         */
        explicit IInteractor();

        /**
         * @brief Processes mouse events.
         * @param mouseState Current mouse state.
         */
        virtual void onMouseEvent(const MouseState& mouseState) = 0;

        /**
         * @brief Processes key events.
         * @param keyState Current key state.
         */
        virtual void onKeyEvent(const KeyState& keyState) = 0;

        /**
         * @brief Sets the camera that is controlled by this interactor.
         * @param camera Camera to control.
         */
        void setCamera(std::shared_ptr<Camera> camera);

        /**
         * @brief Gets the camera that is controlled by this interactor.
         * @return Camera that is controlled.
         */
        std::shared_ptr<Camera> getCamera();

        /**
         * @brief Event for processing mouse events.
         */
        TEvent<IInteractor, MouseState> mouseEvent;

        /**
         * @brief Event for processing key events.
         */
        TEvent<IInteractor, KeyState> keyEvent;

        /**
         * @brief Flag that turns the interactor on and off.
         */
        bool active;

    protected:
        /**
         * @brief Camera to update
         */
        std::shared_ptr<Camera> mCamera;
    };
}
