#include <vislab/graphics/camera.hpp>
#include <vislab/graphics/key_state.hpp>
#include <vislab/graphics/mouse_state.hpp>
#include <vislab/graphics/trackball_interactor.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    TrackballInteractor::TrackballInteractor()
        : Concrete<TrackballInteractor, IInteractor>()
        , mLastX(0)
        , mLastY(0)
        , movementSpeed(0.1)
        , rotationSpeed(0.001)
    {
    }

    void TrackballInteractor::onMouseEvent(const MouseState& mouseState)
    {
        if (mCamera == nullptr)
            return;
        int lastX = mLastX;
        int lastY = mLastY;
        if (mouseState.scrollDelta != 0)
        {
            double y               = mouseState.scrollDelta;
            Eigen::Vector3d lookAt = mCamera->getLookAt();
            Eigen::Vector3d eye    = mCamera->getPosition();
            Eigen::Vector3d up     = mCamera->getUp();
            Eigen::Vector3d dir    = (lookAt - eye).normalized();
            if (y != 0)
            {
                eye = eye + dir * y;
            }
            mCamera->setPosition(eye);
        }
        if (mouseState.leftIsDown && !mouseState.leftDown)
        {
            double rx = (mouseState.x - lastX) * rotationSpeed;
            double ry = -(mouseState.y - lastY) * rotationSpeed;
            if (rx != 0 || ry != 0)
            {
                Eigen::Vector3d lookAt = mCamera->getLookAt();
                Eigen::Vector3d eye    = mCamera->getPosition();
                Eigen::Vector3d up     = mCamera->getUp();

                // rotate left - right
                Eigen::Vector3d dir    = lookAt - eye;
                double distance        = dir.length();
                Eigen::Quaterniond rot = Eigen::Quaterniond(Eigen::AngleAxisd(rx, up));
                dir                    = rot * dir;

                // rotate up - down
                Eigen::Vector3d lastDirection = dir;
                Eigen::Vector3d right         = dir.cross(up).normalized();
                rot                           = Eigen::Quaterniond(Eigen::AngleAxisd(ry, right));
                dir                           = rot * dir;

                // prevent flipping around the up-vector
                if (std::abs(dir.normalized().dot(up)) > 0.99)
                    dir = lastDirection;

                // lookAt = eye + dir;
                eye = (lookAt - dir.normalized() * distance);
                mCamera->setLookAt(lookAt);
                mCamera->setPosition(eye);
            }
        }
        if (mouseState.middleIsDown && !mouseState.middleDown)
        {
            Eigen::Vector3d lookAt = mCamera->getLookAt();
            Eigen::Vector3d eye    = mCamera->getPosition();
            double scalefactor     = (eye - lookAt).length() * 0.01;
            double mx              = (mouseState.x - lastX) * movementSpeed * scalefactor;
            double my              = (mouseState.y - lastY) * movementSpeed * scalefactor;
            if (mx != 0 || my != 0)
            {
                Eigen::Vector3d up    = mCamera->getUp();
                Eigen::Vector3d dir   = lookAt - eye;
                Eigen::Vector3d right = dir.cross(up).normalized();
                Eigen::Vector3d inup  = right.cross(dir).normalized();
                lookAt += right * mx + inup * my;
                eye += right * mx + inup * my;
                mCamera->setLookAt(lookAt);
                mCamera->setPosition(eye);
            }
        }
        mLastX = mouseState.x;
        mLastY = mouseState.y;
        mouseEvent.notify(this, &mouseState);
    }

    void TrackballInteractor::onKeyEvent(const KeyState& keyState)
    {
        double y = ((keyState.isDown_W ? 1 : 0) - (keyState.isDown_S ? 1 : 0)) * movementSpeed;
        if (y != 0)
        {
            Eigen::Vector3d lookAt = mCamera->getLookAt();
            Eigen::Vector3d eye    = mCamera->getPosition();
            Eigen::Vector3d dir    = (lookAt - eye).normalized();
            if (y != 0)
            {
                eye = eye + dir * y;
            }
            mCamera->setPosition(eye);
        }
        keyEvent.notify(this, &keyState);
    }

    void TrackballInteractor::serialize(IArchive& archive)
    {
        archive("LastX", mLastX);
        archive("LastY", mLastY);
        archive("MovementSpeed", movementSpeed);
        archive("RotationSpeed", rotationSpeed);
    }
}
