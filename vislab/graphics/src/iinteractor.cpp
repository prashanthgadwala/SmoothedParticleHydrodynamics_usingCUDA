#include <vislab/graphics/iinteractor.hpp>

#include <vislab/graphics/camera.hpp>

namespace vislab
{
    IInteractor::IInteractor()
        : mCamera(nullptr)
        , active(true)
    {
    }

    void IInteractor::setCamera(std::shared_ptr<Camera> camera) { mCamera = camera; }
    std::shared_ptr<Camera> IInteractor::getCamera() { return mCamera; }
}
