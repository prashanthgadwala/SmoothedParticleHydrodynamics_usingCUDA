#include <vislab/graphics/projective_camera.hpp>

//#include <vislab/graphics/interaction.hpp>
//#include <vislab/graphics/ray.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    ProjectiveCamera::ProjectiveCamera()
        : mProj(Eigen::Matrix4d::Identity())
        , mNear(0.01)
        , mFar(1.0)
        , mWidth(1.0)
        , mHeight(1.0)
    {
    }

    const Eigen::Matrix4d& ProjectiveCamera::getProj() const { return mProj; }
    double ProjectiveCamera::getNear() const { return mNear; }
    double ProjectiveCamera::getFar() const { return mFar; }
    double ProjectiveCamera::getWidth() const { return mWidth; }
    double ProjectiveCamera::getHeight() const { return mHeight; }
    double ProjectiveCamera::getAspectRatio() const { return mWidth / mHeight; }

    void ProjectiveCamera::setNear(double near)
    {
        if (mNear != near)
        {
            mNear = near;
            updateProjMatrix();
        }
    }

    void ProjectiveCamera::setFar(double far)
    {
        if (mFar != far)
        {
            mFar = far;
            updateProjMatrix();
        }
    }

    void ProjectiveCamera::setWidth(double width)
    {
        if (mWidth != width)
        {
            mWidth = width;
            updateProjMatrix();
        }
    }

    void ProjectiveCamera::setHeight(double height)
    {
        if (mHeight != height)
        {
            mHeight = height;
            updateProjMatrix();
        }
    }

    Eigen::Vector2d ProjectiveCamera::project(const Eigen::Vector3d& coord) const
    {
        Eigen::Vector4d p   = Eigen::Vector4d(coord.x(), coord.y(), coord.z(), 1);
        Eigen::Vector4d npc = mProj * (mView * p);
        npc /= npc.w();
        return Eigen::Vector2d(
            (npc.x() * 0.5 + 0.5) * mWidth,
            (-npc.y() * 0.5 + 0.5) * mHeight);
    }

    Eigen::Vector3d ProjectiveCamera::unproject(const Eigen::Vector3d& coord) const
    {
        const double viewport_minx = 0;
        const double viewport_miny = 0;
        Eigen::Matrix4d matrix     = mProj * mView;
        Eigen::Vector3d v(
            (((coord.x() - viewport_minx) / mWidth) * 2.0) - 1.0,
            -((((coord.y() - viewport_miny) / mHeight) * 2.0) - 1.0),
            (coord.z() - mNear) / (mFar - mNear));
        Eigen::Vector4d vv = matrix.inverse() * Eigen::Vector4d(v.x(), v.y(), v.z(), 1);
        return vv.xyz() / vv.w();
    }

    void ProjectiveCamera::serialize(IArchive& archive)
    {
        Camera::serialize(archive);
        archive("Near", mNear);
        archive("Far", mFar);
        archive("Width", mWidth);
        archive("Height", mHeight);
        archive("Proj", mProj);
    }
}
