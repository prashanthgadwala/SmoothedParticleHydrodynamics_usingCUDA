#pragma once

#include "component.hpp"

#include <Eigen/Eigen>

namespace vislab
{
    /**
     * @brief Class that implements a basic camera.
     */
    class Camera : public Interface<Camera, Component>
    {
    public:
        /**
         * @brief Constructor.
         */
        explicit Camera();

        /**
         * @brief Gets the view matrix.
         * @return View matrix.
         */
        [[nodiscard]] const Eigen::Matrix4d& getView() const;

        /**
         * @brief Gets the position of the camera.
         * @return Camera position.
         */
        [[nodiscard]] const Eigen::Vector3d& getPosition() const;

        /**
         * @brief Gets the location that the camera looks at.
         * @return Look at location.
         */
        [[nodiscard]] const Eigen::Vector3d& getLookAt() const;

        /**
         * @brief Gets the up vector of the camera.
         * @return Up vector.
         */
        [[nodiscard]] const Eigen::Vector3d& getUp() const;

        /**
         * @brief Sets the position of the camera.
         * @param Position Camera position.
         */
        void setPosition(const Eigen::Vector3d& Position);

        /**
         * @brief Sets the location that the camera looks at.
         * @param LookAt Look at location.
         */
        void setLookAt(const Eigen::Vector3d& LookAt);

        /**
         * @brief Sets the up vector of the camera.
         * @param Up Up vector.
         */
        void setUp(const Eigen::Vector3d& Up);

        /**
         * @brief Transforms all corners of a world space bounding box to view space and returns the view space bounding box of the transformed corners.
         * @param box World space bounding box to transform.
         * @return View space bounding box.
         */
        [[nodiscard]] Eigen::AlignedBox3d getViewspaceAABB(const Eigen::AlignedBox3d& box) const;

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

    protected:
        /**
         * @brief Recomputes the view matrix.
         */
        void updateViewMatrix();

        /**
         * @brief View matrix.
         */
        Eigen::Matrix4d mView;

        /**
         * @brief Position of the camera.
         */
        Eigen::Vector3d mPosition;

        /**
         * @brief Location that the camera looks at.
         */
        Eigen::Vector3d mLookAt;

        /**
         * @brief Up vector of the camera.
         */
        Eigen::Vector3d mUp;
    };
}
