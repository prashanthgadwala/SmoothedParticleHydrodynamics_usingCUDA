#include <vislab/graphics/camera.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    Camera::Camera()
        : mPosition(1, 0, 0)
        , mLookAt(0, 0, 0)
        , mUp(0, 0, 1)
        , mView(Eigen::Matrix4d::Identity())
    {
        updateViewMatrix();
    }

    const Eigen::Matrix4d& Camera::getView() const { return mView; }
    const Eigen::Vector3d& Camera::getPosition() const { return mPosition; }
    const Eigen::Vector3d& Camera::getLookAt() const { return mLookAt; }
    const Eigen::Vector3d& Camera::getUp() const { return mUp; }

    void Camera::setPosition(const Eigen::Vector3d& Position)
    {
        mPosition = Position;
        updateViewMatrix();
    }

    void Camera::setLookAt(const Eigen::Vector3d& LookAt)
    {
        mLookAt = LookAt;
        updateViewMatrix();
    }

    void Camera::setUp(const Eigen::Vector3d& Up)
    {
        mUp = Up;
        updateViewMatrix();
    }

    Eigen::AlignedBox3d Camera::getViewspaceAABB(const Eigen::AlignedBox3d& box) const
    {
        Eigen::AlignedBox3d resBox;
        resBox.setEmpty();
        for (int i = 0; i < 8; ++i)
        {
            Eigen::Vector3d corner    = box.corner((Eigen::AlignedBox3d::CornerType)i);
            Eigen::Vector4d vrcCorner = mView * Eigen::Vector4d(corner.x(), corner.y(), corner.z(), 1);
            vrcCorner /= vrcCorner.w();
            resBox.extend(vrcCorner.xyz());
        }
        return resBox;
    }

    void Camera::serialize(IArchive& archive)
    {
        archive("Position", mPosition);
        archive("LookAt", mLookAt);
        archive("Up", mUp);
        archive("View", mView);
    }

    void Camera::updateViewMatrix()
    {
        mView = Eigen::Matrix4d::lookAtLH(mPosition, mLookAt, mUp);
        markChanged();
    }
}
