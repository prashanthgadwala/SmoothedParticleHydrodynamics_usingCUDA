#pragma once

#include "light_gl.hpp"

namespace vislab
{
    class PointLight;

    /**
     * @brief Point light source.
     */
    class PointLightGl : public Concrete<PointLightGl, LightGl>
    {
    public:
        /**
         * @brief Constructor.
         */
        PointLightGl(std::shared_ptr<const PointLight> pointLight);

        /**
         * @brief Destructor. Releases the opengl resources.
         */
        virtual ~PointLightGl();

        /**
         * @brief Updates the content of the light.
         * @param scene Scene that holds the wrappers to the different resources.
         * @param opengl Reference to the openGL handle.
         */
        void update(SceneGl* scene, OpenGL* opengl) override;

        /**
         * @brief Creates the device resources.
         * @param opengl Reference to the openGL handle.
         * @return True, if the creation succeeded.
         */
        [[nodiscard]] bool createDevice(OpenGL* opengl) override;

        /**
         * @brief Releases the device resources.
         */
        void releaseDevice() override;

        /**
         * @brief Gets access to the underlying component that is wrapped.
         * @return Pointer to the underlying component.
         */
        [[nodiscard]] const PointLight* get() const;

        /**
         * @brief Returns the code for sampling a light source.
         * @return String containing the code.
         */
        static std::string generateCode() noexcept;
    };
}
