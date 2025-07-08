#pragma once

#include <vislab/opengl/constant_buffer_gl.hpp>
#include <vislab/opengl/renderer_gl.hpp>

#include <Eigen/Eigen>

#include <memory>
#include <unordered_map>

namespace vislab
{
    class Scene;
    class OpenGL;
    class ShaderGl;
    // class Mesh;
    // class ColormapTexture;
    class Camera;
    class SceneGl;

    /**
     * @brief Basic OpenGL renderer for the physsim course.
     */
    class ForwardRendererGl : public Concrete<ForwardRendererGl, RendererGl>
    {
    public:
        /**
         * @brief Constructor.
         */
        ForwardRendererGl();

        /**
         * @brief Destructor. Releases the opengl resources.
         */
        virtual ~ForwardRendererGl();

        /**
         * @brief Renders the scene from the given camera.
         */
        void render();

        /**
         * @brief Renders the scene for the given camera.
         */
        void render(ProgressInfo& progressInfo) override;

        /**
         * @brief Creates the device resources.
         * @return True, if the creation succeeded.
         */
        [[nodiscard]] bool createDevice();

        /**
         * @brief Creates the swap chain resources, i.e., resources that depend on the screen resolution.
         * @return True, if the creation succeeded.
         */
        [[nodiscard]] bool createSwapChain();

        /**
         * @brief Releases the device resources.
         */
        void releaseDevice();

        /**
         * @brief Releases the swap chain resources.
         */
        void releaseSwapChain();

        ///**
        // * @brief Draws the frame by invoking OpenGL calls.
        // * @param opengl Reference to the OpenGL context.
        // */
        // void draw(OpenGL* opengl);

        ///**
        // * @brief Clears resources that are stored for a certain mesh.
        // * @param mesh Mesh to clear resources from.
        // */
        // void clearMeshResources(Mesh* mesh);

        ///**
        // * @brief Clears resources that are stored for a certain texture.
        // * @param texture Texture to clear resources from.
        // */
        // void clearTextureResources(ColormapTexture* texture);

    protected:
        /**
         * @brief Uniform buffer parameters.
         */
        struct LightParams
        {
            /**
             * @brief Point light positions.
             */
            Eigen::Vector4f lightPos[8];

            /**
             * @brief Point light intensities.
             */
            Eigen::Vector4f lightIntensity[8];
        };

        /**
         * @brief Wrapper that holds a uniform buffer.
         */
        ConstantBufferGl<LightParams> mLightParams;

        ///**
        // * @brief Shader program for the rendering of spheres.
        // */
        // std::shared_ptr<ShaderGl> mSphereShader;

        ///**
        // * @brief Shader program for the rendering of rectangles.
        // */
        // std::shared_ptr<ShaderGl> mRectangleShader;

        ///**
        // * @brief Shader program for the rendering of rectangles with texture.
        // */
        // std::shared_ptr<ShaderGl> mTexturedRectangleShader;

        ///**
        // * @brief Shader program for the rendering of triangles with flat shading.
        // */
        // std::shared_ptr<ShaderGl> mFlatTriangleShader;

        ///**
        // * @brief Shader program for the rendering of triangles with phong shading.
        // */
        // std::shared_ptr<ShaderGl> mTriangleShader;

        ///**
        // * @brief Vertex array object for the sphere.
        // */
        // uint32_t mSphereVAO;

        ///**
        // * @brief Vertex array object for the rectangle.
        // */
        // uint32_t mRectangleVAO;

        ///**
        // * @brief Collection of OpenGL handles related to a triangle mesh.
        // */
        // struct MeshResource
        //{
        //    /**
        //     * @brief Vertex buffer of positions.
        //     */
        //    unsigned int vbo_positions;

        //    /**
        //     * @brief Vertex buffer of normals.
        //     */
        //    unsigned int vbo_normals;

        //    /**
        //     * @brief Index buffer.
        //     */
        //    unsigned int ibo;

        //    /**
        //     * @brief Vertex array object.
        //     */
        //    unsigned int vao;
        //};

        ///**
        // * @brief Helper function that does lazy construction of VBOs, IBOs, and VAOs from a mesh.
        // * @param mesh Mesh to map to GL resources.
        // * @return Handle to the mesh resources.
        // */
        // MeshResource getMeshResources(vislab::Mesh* mesh);

        ///**
        // * @brief Collection of VBOs IBOs, and VAOs for the triangle meshes.
        // */
        // std::unordered_map<vislab::Mesh*, MeshResource> mTriangleMeshes;

        ///**
        // * @brief Helper function that does lazy construction of textures.
        // * @param texture Color mapped texture to build GL resources for.
        // * @return Handle to texture.
        // */
        // unsigned int getTextureResources(vislab::ColormapTexture* texture);

        ///**
        // * @brief Collection of texture handles for color mapped textures.
        // */
        // std::unordered_map<vislab::ColormapTexture*, unsigned int> mTextures;

    private:
        /**
         * @brief Generates the shader code that is injected in shader compilations.
         * @return Shader code.
         */
        static std::vector<ShaderSourceGl> generateShaderCodes();
    };
}
