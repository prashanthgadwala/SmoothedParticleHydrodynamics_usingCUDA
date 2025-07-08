#pragma once

#include "resource_gl.hpp"

#include <vislab/graphics/spectrum.hpp>

#include <Eigen/Eigen>

namespace vislab
{
    class BSDF;

    /**
     * @brief Base class for bidirectional scattering distribution functions.
     */
    class BSDFGl : public Interface<BSDFGl, ResourceGl<BSDF>>
    {
    public:
        /**
         * @brief Constructor.
         * @param bsdf BSDF to be wrapped.
         */
        BSDFGl(std::shared_ptr<const BSDF> bsdf, const std::string& shaderCode);

        /**
         * @brief Binds the constant buffer for rendering.
         * @param shaderProgram Shader program handle to bind the parameters to.
         * @param textureBindingPoint Binding point to bind texture to.
         * @param keyBindingPoint Binding point to bind shader storage buffer with keys to.
         * @param valuesBindingPoint Binding point to bind shader storage buffer with values to.
         * @param cbBindingPointTf Binding point to bind constant buffer for the transfer function to.
         * @param cbBindingPointBsdf Binding point to bind constant buffer for the bsdf parameters to.
         * @return True, if binding was successful.
         */
        [[nodiscard]] virtual bool bind(unsigned int shaderProgram, int textureBindingPoint, int keyBindingPoint, int valuesBindingPoint, int cbBindingPointTf, int cbBindingPointBsdf) = 0;

        /**
         * @brief Generates the shader code for working with this object.
         * @return String containing the generated shader code.
         */
        [[nodiscard]] const std::string& getShaderCode() const;

    private:
        /**
         * @brief Shader code for evaluating the BSDF.
         */
        std::string mShaderCode;
    };
}
