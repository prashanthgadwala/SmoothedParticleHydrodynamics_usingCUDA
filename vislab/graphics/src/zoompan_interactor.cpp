#include <vislab/graphics/zoompan_interactor.hpp>

#include <vislab/graphics/key_state.hpp>
#include <vislab/graphics/mouse_state.hpp>
#include <vislab/graphics/projective_camera.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    ZoomPanInteractor::ZoomPanInteractor()
        : Concrete<ZoomPanInteractor, IInteractor>()
        , mLastX(0)
        , mLastY(0)
        , movementSpeed(1)
    {
    }

    void ZoomPanInteractor::onMouseEvent(const MouseState& mouseState)
    {
        auto projectiveCamera = std::dynamic_pointer_cast<ProjectiveCamera>(mCamera);
        if (projectiveCamera == nullptr)
            return;

        int lastX = mLastX;
        int lastY = mLastY;
        if (mouseState.scrollDelta != 0)
        {
            double y = mouseState.scrollDelta;
            projectiveCamera->setWidth(projectiveCamera->getWidth() - y);
        }
        if (mouseState.middleIsDown && !mouseState.middleDown)
        {
            double scalefactor = projectiveCamera->getWidth();
            double mx          = (mouseState.x - lastX) * movementSpeed * scalefactor;
            double my          = (mouseState.y - lastY) * movementSpeed * scalefactor;
            if (mx != 0 || my != 0)
            {
                Eigen::Vector3d lookAt = projectiveCamera->getLookAt();
                Eigen::Vector3d eye    = projectiveCamera->getPosition();
                Eigen::Vector3d up     = projectiveCamera->getUp();
                Eigen::Vector3d dir    = lookAt - eye;
                Eigen::Vector3d right  = dir.cross(up).normalized();
                Eigen::Vector3d inup   = right.cross(dir).normalized();
                lookAt += right * mx + inup * my;
                eye += right * mx + inup * my;
                projectiveCamera->setLookAt(lookAt);
                projectiveCamera->setPosition(eye);
            }
        }
        mLastX = mouseState.x;
        mLastY = mouseState.y;
        mouseEvent.notify(this, &mouseState);
    }

    void ZoomPanInteractor::onKeyEvent(const KeyState& keyState)
    {
        keyEvent.notify(this, &keyState);
    }

    void ZoomPanInteractor::serialize(IArchive& archive)
    {
        archive("LastX", mLastX);
        archive("LastY", mLastY);
        archive("MovementSpeed", movementSpeed);
    }
}
