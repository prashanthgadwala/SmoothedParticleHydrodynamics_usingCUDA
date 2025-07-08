#include <vislab/graphics/transform.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    Transform::Transform()
        : mMatrix(Eigen::Matrix4d::Identity())
        , mMatrixInverse(Eigen::Matrix4d::Identity())
    {
    }
    Transform::Transform(const Eigen::Matrix4d& matrix)
        : mMatrix(matrix)
        , mMatrixInverse(matrix.inverse())
    {
    }

    Transform::Transform(const Eigen::Vector3d& translation, const Eigen::Quaterniond& rotation, const Eigen::Vector3d& scale)
    {
        mMatrix       = Eigen::Matrix4d::diag(scale);
        mMatrix(3, 3) = 1.0;
        mMatrix.block(0, 0, 3, 3) *= rotation.toRotationMatrix();
        mMatrix.block(0, 3, 3, 1) = translation;
        mMatrixInverse            = mMatrix.inverse();
    }

    const Eigen::Matrix4d& Transform::getMatrix() const { return mMatrix; }
    const Eigen::Matrix4d& Transform::getMatrixInverse() const { return mMatrixInverse; }

    void Transform::setMatrix(const Eigen::Matrix4d& matrix)
    {
        mMatrix        = matrix;
        mMatrixInverse = matrix.inverse();
        markChanged();
        onChanged.notify(this, &mMatrix);
    }

    void Transform::serialize(IArchive& archive)
    {
        archive("Matrix", mMatrix);
        archive("MatrixInverse", mMatrixInverse);
    }

    Eigen::Vector3d Transform::transformPoint(const Eigen::Vector3d& point) const
    {
        Eigen::Vector4d p4 = mMatrix * Eigen::Vector4d(point.x(), point.y(), point.z(), 1.);
        return p4.xyz() / p4.w();
    }

    Eigen::Vector3d Transform::transformVector(const Eigen::Vector3d& direction) const
    {
        return mMatrix.block(0, 0, 3, 3) * direction;
    }

    Eigen::Vector3d Transform::transformNormal(const Eigen::Vector3d& normal) const
    {
        return (mMatrixInverse.block(0, 0, 3, 3).transpose() * normal).stableNormalized();
    }

    /*Ray3d Transform::transformRay(const Ray3d& ray) const
    {
        Ray3d result     = ray;
        result.origin    = transformPoint(ray.origin);
        result.direction = transformVector(ray.direction);
        return result;
    }*/

    Eigen::Vector3d Transform::transformPointInverse(const Eigen::Vector3d& point) const
    {
        Eigen::Vector4d p4 = mMatrixInverse * Eigen::Vector4d(point.x(), point.y(), point.z(), 1.);
        return p4.xyz() / p4.w();
    }

    Eigen::Vector3d Transform::transformVectorInverse(const Eigen::Vector3d& direction) const
    {
        return mMatrixInverse.block(0, 0, 3, 3) * direction;
    }

    Eigen::AlignedBox3d Transform::transformBox(const Eigen::AlignedBox3d& oob) const
    {
        Eigen::AlignedBox3d aabb;
        aabb.setEmpty();
        aabb.extend(transformPoint(oob.corner(Eigen::AlignedBox3d::CornerType::BottomLeftFloor)));
        aabb.extend(transformPoint(oob.corner(Eigen::AlignedBox3d::CornerType::BottomRightFloor)));
        aabb.extend(transformPoint(oob.corner(Eigen::AlignedBox3d::CornerType::TopLeftFloor)));
        aabb.extend(transformPoint(oob.corner(Eigen::AlignedBox3d::CornerType::TopRightFloor)));
        aabb.extend(transformPoint(oob.corner(Eigen::AlignedBox3d::CornerType::BottomLeftCeil)));
        aabb.extend(transformPoint(oob.corner(Eigen::AlignedBox3d::CornerType::BottomRightCeil)));
        aabb.extend(transformPoint(oob.corner(Eigen::AlignedBox3d::CornerType::TopLeftCeil)));
        aabb.extend(transformPoint(oob.corner(Eigen::AlignedBox3d::CornerType::TopRightCeil)));
        return aabb;
    }
}
