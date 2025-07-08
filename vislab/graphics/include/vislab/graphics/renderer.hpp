#pragma once

#include <vislab/core/object.hpp>
#include <vislab/core/progress_info.hpp>

#include <future>

namespace vislab
{
    class Scene;
    class Camera;

    /**
     * @brief Base class for renderers.
     */
    class Renderer : public Interface<Renderer, Object>
    {
    public:
        /**
         * @brief Renders the scene from the given camera.
         */
        void render();

        /**
         * @brief Renders the scene from the given camera.
         * @param progress Can be used to monitor the progress.
         */
        virtual void render(ProgressInfo& progress) = 0;

        /**
         * @brief Renders the scene from the given camera asynchronously.
         * @param progress Can be used to monitor the progress.
         */
        std::future<void> renderAsync(ProgressInfo& progress);

        /**
         * @brief Update the rendering resources for the scene.
         */
        virtual void update() = 0;

        /**
         * @brief Scene to render.
         */
        std::shared_ptr<const Scene> scene;

        /**
         * @brief Active camera to render the scene from. Note that the camera also needs to be attached to an actor in the scene, in order to make sure that its resources are created.
         */
        std::shared_ptr<const Camera> camera;
    };
}
