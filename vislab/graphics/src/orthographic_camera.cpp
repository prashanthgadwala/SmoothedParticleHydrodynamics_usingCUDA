#include <vislab/graphics/orthographic_camera.hpp>

namespace vislab
{
    OrthographicCamera::OrthographicCamera()
    {
        updateProjMatrix();
    }

    void OrthographicCamera::updateProjMatrix()
    {
        mProj = Eigen::Matrix3d::orthoLH(mWidth, mHeight, mNear, mFar);
        markChanged();
    }
}
