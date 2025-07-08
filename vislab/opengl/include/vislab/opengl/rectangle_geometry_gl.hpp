#pragma once

#include "geometry_gl.hpp"

namespace vislab
{
    class RectangleGeometry;

    /**
     * @brief Class that defines a rectangle spanning (-1,-1,0) to (1,1,0). To modify its position update the transformation.
     */
    class RectangleGeometryGl : public Concrete<RectangleGeometryGl, GeometryGl>
    {
    public:
        /**
         * @brief Constructor.
         */
        RectangleGeometryGl(std::shared_ptr<const RectangleGeometry> rectangleGeometry);

        /**
         * @brief Destructor. Releases the opengl resources.
         */
        virtual ~RectangleGeometryGl();

        /**
         * @brief Updates the content of the geometry.
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
         * @brief Draws the geometry.
         */
        void draw() override;

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
        [[nodiscard]] const RectangleGeometry* get() const;

    private:
        /**
         * @brief Generates the shader code that is to be injected during compilation.
         * @return Shader source code.
         */
        static ShaderSourceGl generateSourceCode();

        /**
         * @brief Vertex array object.
         */
        unsigned int mVAO;
    };
}
