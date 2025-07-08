#pragma once

#include <vislab/core/object.hpp>

#include "resource_gl.hpp"

#include <vislab/graphics/spectrum.hpp>

namespace vislab
{
    class Actor;
    class BSDFGl;
    class LightGl;
    class GeometryGl;
    class TransformGl;
    class SceneGl;
    class CameraGl;
    class RendererGl;
    class ShaderGl;
    class ShaderSourceGl;

    /**
     * @brief Actor that can be placed in the scene.
     */
    class ActorGl : public Concrete<ActorGl, ResourceGl<Actor>>
    {
    public:
        /**
         * @brief Constructor.
         */
        ActorGl(std::shared_ptr<const Actor> actor);

        /**
         * @brief Updates the content of the actor.
         * @param scene Scene that holds the wrappers to the different resources.
         * @param renderer Renderer that the actor is built for (the renderer can supply shader code).
         * @param opengl Reference to the openGL handle.
         */
        void update(SceneGl* scene, const RendererGl* renderer, OpenGL* opengl);

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
         * @brief Bounding box the object in world space.
         * @return World-space bounding box.
         */
        [[nodiscard]] Eigen::AlignedBox3d worldBounds() const;

        /**
         * @brief Optional BSDF of the actor.
         */
        std::shared_ptr<BSDFGl> bsdf;

        /**
         * @brief Optional light source of the actor.
         */
        std::shared_ptr<LightGl> light;

        /**
         * @brief Optional geometry of the actor.
         */
        std::shared_ptr<GeometryGl> geometry;

        /**
         * @brief Transformation
         */
        std::shared_ptr<TransformGl> transform;

        /**
         * @brief Optional camera.
         */
        std::shared_ptr<CameraGl> camera;

        /**
         * @brief Map of shaders for the actors, indexed by the renderer UID.
         */
        std::unordered_map<std::size_t, std::shared_ptr<ShaderGl>> shaders;

    private:
        /**
         * @brief Assembles the shader code.
         * @param rendererShaderCode Rendering code given by the renderer.
         * @param cameraShaderCode Rendering code given by the camera.
         */
        void assembleShaderCode(std::size_t uid, const ShaderSourceGl& rendererShaderCode, const std::string& cameraShaderCode);
    };
}
