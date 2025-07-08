#pragma once

#include "resource_gl.hpp"

namespace vislab
{
    class Camera;

    /**
     * @brief Interface for cameras.
     */
    class CameraGl : public Interface<CameraGl, ResourceGl<Camera>>
    {
    public:
        /**
         * @brief Constructor.
         * @param camera Camera to be wrapped.
         * @param shaderCode Camera shader code that is injected in the shader compilation.
         */
        CameraGl(std::shared_ptr<const Camera> camera, const std::string& shaderCode);

        /**
         * @brief Binds the constant buffer for rendering.
         * @param shaderProgram Shader program handle to bind the parameters to.
         * @param bindingPoint Binding point to bind to.
         * @return True, if binding was successful.
         */
        [[nodiscard]] virtual bool bind(unsigned int shaderProgram, int bindingPoint) = 0;

        /**
         * @brief Generates the shader code for working with this object.
         * @return String containing the generated shader code.
         */
        [[nodiscard]] const std::string& getShaderCode() const;

    private:
        /**
         * @brief Shader code for declaring the transformation parameters.
         */
        std::string mShaderCode;
    };
}
