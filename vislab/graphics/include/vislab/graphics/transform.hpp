#pragma once

// #include "ray.hpp"

#include "component.hpp"

#include "tag.hpp"

#include <vislab/core/event.hpp>

#include <Eigen/Eigen>

namespace vislab
{
    /**
     * @brief Component that stores a transformation.
     */
    class Transform : public Concrete<Transform, Component>
    {
    public:
        /**
         * @brief Constructor with an identity transformation.
         */
        Transform();

        /**
         * @brief Constructor with a given initial matrix.
         * @param matrix Initial matrix.
         */
        Transform(const Eigen::Matrix4d& matrix);

        /**
         * @brief Constructs a transformation from a scale, rotation, and translation (in that order).
         * @param translation Translation vector.
         * @param rotation Rotation quaternion.
         * @param scale Scaling vector.
        */
        Transform(const Eigen::Vector3d& translation, const Eigen::Quaterniond& rotation = Eigen::Quaterniond::Identity(), const Eigen::Vector3d& scale = Eigen::Vector3d::Ones());

        /**
         * @brief Gets the transformation matrix.
         * @return Transformation matrix.
         */
        [[nodiscard]] const Eigen::Matrix4d& getMatrix() const;

        /**
         * @brief Inverse of the transformation matrix.
         * @return Inverse matrix.
         */
        [[nodiscard]] const Eigen::Matrix4d& getMatrixInverse() const;

        /**
         * @brief Sets the world transformation matrix.
         * @param matrix Transformation matrix to set.
         */
        void setMatrix(const Eigen::Matrix4d& matrix);

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Event that is raised when the matrix changed.
         */
        TEvent<Transform, Eigen::Matrix4d> onChanged;

        /**
         * @brief Affine transformation of a 3D coordinate.
         * @param point Point to transform.
         * @return Transformed point.
         */
        [[nodiscard]] Eigen::Vector3d transformPoint(const Eigen::Vector3d& point) const;

        /**
         * @brief Transforms a direction vector.
         * @param direction Direction vector to transform.
         * @return Transformed direction vector.
         */
        [[nodiscard]] Eigen::Vector3d transformVector(const Eigen::Vector3d& direction) const;

        /**
         * @brief Transforms a normal vector.
         * @param normal Normal vector to transform.
         * @return Transformed normal vector.
         */
        [[nodiscard]] Eigen::Vector3d transformNormal(const Eigen::Vector3d& normal) const;

        /**
         * @brief Affine inverse transformation of a 3D coordinate.
         * @param point Point to transform.
         * @return Transformed point.
         */
        [[nodiscard]] Eigen::Vector3d transformPointInverse(const Eigen::Vector3d& point) const;

        /**
         * @brief Inverse transform of a direction vector.
         * @param direction Direction vector to transform.
         * @return Transformed direction vector.
         */
        [[nodiscard]] Eigen::Vector3d transformVectorInverse(const Eigen::Vector3d& direction) const;

        /**
         * @brief Transforms the corners of a bounding box and fits an axis aligned bounding box around them.
         * @param oob Object-oriented bounding box to transform.
         * @return Axis-aligned bounding box around the transformed corners of the object-oriented bounding box.
        */
        [[nodiscard]] Eigen::AlignedBox3d transformBox(const Eigen::AlignedBox3d& oob) const;

    private:
        /**
         * @brief Transformation matrix.
         */
        Eigen::Matrix4d mMatrix;

        /**
         * @brief Inverse of transformation matrix.
         */
        Eigen::Matrix4d mMatrixInverse;
    };
}
