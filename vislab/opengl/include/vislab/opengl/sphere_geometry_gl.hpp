#pragma once

#include "geometry_gl.hpp"

#include "constant_buffer_gl.hpp"

namespace vislab
{
    class SphereGeometry;

    /**
     * @brief Class for the rendering of spheres.
     */
    class SphereGeometryGl : public Concrete<SphereGeometryGl, GeometryGl>
    {
    public:
        /**
         * @brief Constructor.
         */
        SphereGeometryGl(std::shared_ptr<const SphereGeometry> sphereGeometry);

        /**
         * @brief Destructor. Releases the opengl resources.
         */
        virtual ~SphereGeometryGl();

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
        [[nodiscard]] const SphereGeometry* get() const;

    private:
        /**
         * @brief Generates the shader code that is to be injected during compilation.
         * @return Shader source code.
         */
        static ShaderSourceGl generateSourceCode();

        /**
         * @brief Uniform buffer parameters.
         */
        struct Params
        {
            /**
             * @brief Scaling factor for the size of the spheres.
             */
            float radiusScale;
            /**
             * @brief Flag that informs whether there is a radius buffer bound.
             */
            int hasRadiusBuffer;
        };

        /**
         * @brief Wrapper that holds a uniform buffer.
         */
        vislab::ConstantBufferGl<Params> mParams;

        /**
         * @brief Vertex buffer object with position of vertices.
         */
        unsigned int mVBO_position;

        /**
         * @brief Vertex buffer object with radius per vertex.
         */
        unsigned int mVBO_radius;

        /**
         * @brief Vertex buffer object with data per vertex.
         */
        unsigned int mVBO_data;

        /**
         * @brief Vertex array object.
         */
        unsigned int mVAO;

        /**
         * @brief Number of vertices to draw.
         */
        unsigned int mNumVertices;
    };
}
