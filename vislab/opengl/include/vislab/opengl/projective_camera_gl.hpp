#pragma once

#include "camera_gl.hpp"

#include <vislab/opengl/constant_buffer_gl.hpp>

#include <Eigen/Eigen>

namespace vislab
{
    class ProjectiveCamera;

    /**
     * @brief Interface for cameras.
     */
    class ProjectiveCameraGl : public Concrete<ProjectiveCameraGl, CameraGl>
    {
    public:
        /**
         * @brief Constructor.
         * @param camera Camera to be wrapped.
         */
        ProjectiveCameraGl(std::shared_ptr<const ProjectiveCamera> camera);

        /**
         * @brief Destructor. Releases the opengl resources.
         */
        virtual ~ProjectiveCameraGl();

        /**
         * @brief Updates this resource.
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
         * @brief Binds the constant buffer for rendering.
         * @param shaderProgram Shader program handle to bind the parameters to.
         * @param bindingPoint Binding point to bind to.
         * @return True, if binding was successful.
         */
        [[nodiscard]] bool bind(unsigned int shaderProgram, int bindingPoint) override;

        /**
         * @brief Gets access to the underlying component that is wrapped.
         * @return Pointer to the underlying component.
         */
        [[nodiscard]] const ProjectiveCamera* get() const;

    private:
        /**
         * @brief Generates the shader code that is injected in shader compilations.
         * @return Shader code.
         */
        static std::string generateShaderCode();

        /**
         * @brief Uniform buffer parameters.
         */
        struct Params
        {
            /**
             * @brief View matrix.
             */
            Eigen::Matrix4f viewMatrix;

            /**
             * @brief Projection matrix.
             */
            Eigen::Matrix4f projMatrix;

            /**
             * @brief Inverse of view projection matrix.
             */
            Eigen::Matrix4f invViewProjMatrix;

            /**
             * @brief World position of the camera.
             */
            Eigen::Vector4f eyePosition;
        };

        /**
         * @brief Wrapper that holds a uniform buffer.
         */
        vislab::ConstantBufferGl<Params> mParams;
    };
}
