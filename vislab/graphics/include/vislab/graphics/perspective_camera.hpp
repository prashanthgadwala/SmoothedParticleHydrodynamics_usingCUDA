#pragma once

#include "projective_camera.hpp"

namespace vislab
{
    /**
     * @brief Class that implements a projective camera.
     */
    class PerspectiveCamera : public Concrete<PerspectiveCamera, ProjectiveCamera>
    {
    public:
        /**
         * @brief Constructor.
         */
        explicit PerspectiveCamera();

        /**
         * @brief Gets the horizontal field of view (in radians) for perspective projection.
         * @return Horizontal field of view (in radians).
         */
        [[nodiscard]] double getHorizontalFieldOfView() const;

        /**
         * @brief Sets the horizontal field of view (in radians) for perspective projection.
         * @param hfov Horizontal field of view (in radians).
         */
        void setHorizontalFieldOfView(double hfov);

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

    private:
        /**
         * @brief Recomputes the projection matrix.
         */
        void updateProjMatrix() override;

        /**
         * @brief Horizontal field of view (in radians) for perspective projection.
         */
        double mHorizontalFieldOfView;
    };
}
