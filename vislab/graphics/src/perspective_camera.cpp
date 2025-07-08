#include <vislab/graphics/perspective_camera.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    PerspectiveCamera::PerspectiveCamera()
        : mHorizontalFieldOfView(EIGEN_PI / 4)
    {
        updateProjMatrix();
    }

    double PerspectiveCamera::getHorizontalFieldOfView() const { return mHorizontalFieldOfView; }

    void PerspectiveCamera::setHorizontalFieldOfView(double hfov)
    {
        if (mHorizontalFieldOfView != hfov)
        {
            mHorizontalFieldOfView = hfov;
            updateProjMatrix();
        }
    }

    void PerspectiveCamera::updateProjMatrix()
    {
        mProj = Eigen::Matrix4d::perspectiveFovLH(mHorizontalFieldOfView, getAspectRatio(), mNear, mFar);
        markChanged();
    }

    void PerspectiveCamera::serialize(IArchive& archive)
    {
        ProjectiveCamera::serialize(archive);
        archive("HorizontalFieldOfView", mHorizontalFieldOfView);
    }
}
