#pragma once

#include "camera.hpp"

namespace vislab
{
    /**
     * @brief Class that implements a projective camera.
     */
    class ProjectiveCamera : public Interface<ProjectiveCamera, Camera>
    {
    public:
        /**
         * @brief Constructor.
         */
        explicit ProjectiveCamera();

        /**
         * @brief Gets the projection matrix.
         * @return Projection matrix.
         */
        [[nodiscard]] const Eigen::Matrix4d& getProj() const;

        /**
         * @brief Gets the near plane distance of the viewing frustum.
         * @return Near plane distance.
         */
        [[nodiscard]] double getNear() const;

        /**
         * @brief Gets the far plane distance of the viewing frustum.
         * @return Far plane distance.
         */
        [[nodiscard]] double getFar() const;

        /**
         * @brief Gets the width of the viewport.
         * @return Width of the viewport.
         */
        [[nodiscard]] double getWidth() const;

        /**
         * @brief Gets the height of the viewport.
         * @return Height of the viewport.
         */
        [[nodiscard]] double getHeight() const;

        /**
         * @brief Gets the aspect ratio of the projection.
         * @return Aspect ratio of the viewport.
         */
        [[nodiscard]] double getAspectRatio() const;

        /**
         * @brief Sets the near plane distance of the viewing frustum.
         * @param  Near plane distance.
         */
        void setNear(double near);

        /**
         * @brief Sets the far plane distance of the viewing frustum.
         * @param  Far plane distance.
         */
        void setFar(double far);

        /**
         * @brief Sets the width of the viewport.
         * @param width Width of the viewport.
         */
        void setWidth(double width);

        /**
         * @brief Sets the height of the viewport.
         * @param height Height of the viewport.
         */
        void setHeight(double height);

        /**
         * @brief Projects a 3D world location into screen space coordinates.
         * @param coord World coordinate to project into screen space.
         * @return Screen space coordinate.
         */
        [[nodiscard]] Eigen::Vector2d project(const Eigen::Vector3d& coord) const;

        /**
         * @brief Projects a screen space coordinate (with depth) back into world space.
         * @param coord Screen space coordinate to unproject into world space.
         * @return World space coordinate.
         */
        [[nodiscard]] Eigen::Vector3d unproject(const Eigen::Vector3d& coord) const;

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

    protected:
        /**
         * @brief Recomputes the projection matrix.
         */
        virtual void updateProjMatrix() = 0;

        /**
         * @brief Near plane distance of the viewing frustum.
         */
        double mNear;

        /**
         * @brief Far plane distance of the viewing frustum.
         */
        double mFar;

        /**
         * @brief Width of the viewport.
         */
        double mWidth;

        /**
         * @brief Height of the viewport.
         */
        double mHeight;

        /**
         * @brief Projection matrix.
         */
        Eigen::Matrix4d mProj;
    };
}
