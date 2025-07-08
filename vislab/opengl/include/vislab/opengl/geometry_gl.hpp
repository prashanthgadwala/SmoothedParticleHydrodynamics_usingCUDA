#pragma once

#include "resource_gl.hpp"
#include "shader_source_gl.hpp"

namespace vislab
{
    class Geometry;

    /**
     * @brief Base class for a geometric object.
     */
    class GeometryGl : public Interface<GeometryGl, ResourceGl<Geometry>>
    {
    public:
        /**
         * @brief Constructor.
         * @param geometry Geometry to be wrapped.
         * @param sourceCode Source code that is injected during compilation.
         * @param writesDepth Flag that signals whether the pixel shader will write depth values.
         */
        GeometryGl(std::shared_ptr<const Geometry> geometry, const ShaderSourceGl& sourceCode, bool writesDepth);

        /**
         * @brief Draws the geometry.
         */
        virtual void draw() = 0;

        /**
         * @brief Binds the constant buffer for rendering.
         * @param shaderProgram Shader program handle to bind the parameters to.
         * @param bindingPoint Binding point to bind to.
         * @return True, if binding was successful.
         */
        [[nodiscard]] virtual bool bind(unsigned int shaderProgram, int bindingPoint) = 0;

        /**
         * @brief Source code for geometry operations that are injected during compilation.
         */
        const ShaderSourceGl sourceCode;

        /**
         * @brief Flag that determines whether this geometry writes the depth buffer in the pixel shader.
         */
        const bool writesDepth;
    };
}
