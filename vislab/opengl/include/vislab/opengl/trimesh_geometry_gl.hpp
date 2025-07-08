#pragma once

#include "geometry_gl.hpp"

#include "constant_buffer_gl.hpp"

namespace vislab
{
    class TrimeshGeometry;

    /**
     * @brief Class that holds a triangle mesh.
     */
    class TrimeshGeometryGl : public Concrete<TrimeshGeometryGl, GeometryGl>
    {
    public:
        /**
         * @brief Constructor.
         * @param trimeshGeometry Geometry to wrap.
         */
        TrimeshGeometryGl(std::shared_ptr<const TrimeshGeometry> trimeshGeometry);

        /**
         * @brief Destructor. Releases the opengl resources.
         */
        virtual ~TrimeshGeometryGl();

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
        [[nodiscard]] const TrimeshGeometry* get() const;

    private:
        /**
         * @brief Generates the shader code that is to be injected during compilation.
         * @return Shader source code.
         */
        static ShaderSourceGl generateSourceCode();

        /**
         * @brief Vertex buffer that stores the vertex positions.
         */
        unsigned int mVBO_positions;

        /**
         * @brief Vertex buffer that stores the vertex normals.
         */
        unsigned int mVBO_normals;

        /**
         * @brief Vertex buffer that stores the texture coordinates.
         */
        unsigned int mVBO_texCoords;

        /**
         * @brief Vertex buffer object with data per vertex.
         */
        unsigned int mVBO_data;

        /**
         * @brief Index buffer.
         */
        unsigned int mIBO;

        /**
         * @brief Vertex array object.
         */
        unsigned int mVAO;

        /**
         * @brief Number of indices.
         */
        unsigned int mNumIndices;
    };
}
