#pragma once

#include "resource_gl.hpp"

#include <vislab/graphics/transform.hpp>
#include <vislab/opengl/constant_buffer_gl.hpp>

namespace vislab
{
    /**
     * @brief Wrapper for a transformation object.
     */
    class TransformGl : public Concrete<TransformGl, ResourceGl<Transform>>
    {
    public:
        /**
         * @brief Constructor.
         * @param transform Transformation to be wrapped.
         */
        TransformGl(std::shared_ptr<const Transform> transform);

        /**
         * @brief Destructor. Releases the opengl resources.
         */
        virtual ~TransformGl();

        /**
         * @brief Updates this resource.
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
        [[nodiscard]] bool bind(unsigned int shaderProgram, int bindingPoint);

        /**
         * @brief Gets the shader code for working with this object.
         * @return String containing the generated shader code.
         */
        [[nodiscard]] std::string getShaderCode() const noexcept;

    private:
        /**
         * @brief Uniform buffer parameters.
         */
        struct Params
        {
            /**
             * @brief Model world matrix.
             */
            Eigen::Matrix4f worldMatrix;

            /**
             * @brief Inverse of model world matrix.
             */
            Eigen::Matrix4f invWorldMatrix;

            /**
             * @brief Matrix for transforming normals to world space.
             */
            Eigen::Matrix4f normalMatrix;
        };

        /**
         * @brief Wrapper that holds a uniform buffer.
         */
        vislab::ConstantBufferGl<Params> mParams;
    };
}
