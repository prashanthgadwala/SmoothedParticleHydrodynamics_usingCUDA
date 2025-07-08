#pragma once

#include "shader_source_gl.hpp"

#include <vislab/core/progress_info.hpp>
#include <vislab/graphics/renderer.hpp>

namespace vislab
{
    class Scene;
    class OpenGL;
    class ShaderGl;
    class Camera;
    class SceneGl;

    /**
     * @brief Basic OpenGL renderer for the physsim course.
     */
    class RendererGl : public Interface<RendererGl, Renderer>
    {
    public:
        /**
         * @brief Constructor.
         * @param shaderCodes Renderer specific shader codes for the rendering passes.
         */
        RendererGl(const std::vector<ShaderSourceGl>& shaderCodes);
        
        /**
         * @brief Destructor.
         */
        virtual ~RendererGl();

        /**
         * @brief Creates the device resources that are needed specifically for this renderer. This routine is not responsible for the construction of actor resources.
         * @return True, if the creation succeeded.
         */
        [[nodiscard]] virtual bool createDevice() = 0;

        /**
         * @brief Creates the swap chain resources, i.e., resources that depend on the screen resolution. This routine is not responsible for the construction of actor resources.
         * @return True, if the creation succeeded.
         */
        [[nodiscard]] virtual bool createSwapChain() = 0;

        /**
         * @brief Releases the device resources. This routine is not responsible for the construction of actor resources.
         */
        virtual void releaseDevice() = 0;

        /**
         * @brief Releases the swap chain resources. This routine is not responsible for the construction of actor resources.
         */
        virtual void releaseSwapChain() = 0;

        /**
         * @brief Updates the renderer for a given scene.
         */
        void update() override;

        /**
         * @brief OpenGL context to use for rendering.
         */
        std::shared_ptr<OpenGL> openGL;

        /**
         * @brief Shader code for the render passes.
         */
        const std::vector<ShaderSourceGl> passes;

        /**
         * @brief Unique identifier of this renderer, which remains the same throughout the lifetime.
         */
        const unsigned long long uid;

    protected:
        /**
         * @brief Ray tracing resources for the scene.
         */
        std::shared_ptr<SceneGl> mScene;
    };
}
