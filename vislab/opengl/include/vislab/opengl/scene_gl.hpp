#pragma once

#include <vislab/core/object.hpp>
#include <vislab/graphics/resource.hpp>
#include <vislab/graphics/spectrum.hpp>

#include <unordered_map>

namespace vislab
{
    class Scene;
    class ResourceBaseGl;
    class ActorGl;
    class OpenGL;
    class RendererGl;

    /**
     * @brief Scene for the OpenGL renderer.
     */
    class SceneGl : public Concrete<SceneGl, Object>
    {
    public:
        /**
         * @brief Updates the internal data structures and prepares the scene for rendering.
         * @param opengl Reference to the openGL handle.
         * @param scene Scene that is wrapped.
         * @param renderer Renderer that is rendering the scene.
         */
        void update(OpenGL* opengl, const Scene* scene, const RendererGl* renderer);

        /**
         * @brief Environment light
         */
        // std::shared_ptr<const LightRt> environment;

        /**
         * @brief Gets the wrapped resource for a certain UID object.
         * @tparam TResourceGl Wrapped type of the resource.
         * @param object Resource with unique ID.
         * @param renderer Renderer that the actor is built for.
         * @return Pointer to the wrapped resource and a flag that says whether a new wrapper was allocated.
         */
        template <typename TResourceGl>
        [[nodiscard]] std::pair<std::shared_ptr<TResourceGl>, bool> getActor(std::shared_ptr<const Resource> object, const RendererGl* renderer, OpenGL* opengl)
        {
            // try to find the resource
            auto it = mResources.find(object->uid);

            // does it already exist? return it!
            if (it != mResources.end())
                return std::make_pair(std::static_pointer_cast<TResourceGl>(it->second), false);

            // if it does not exist, then allocate it.
            auto res = allocateActor(object, renderer, opengl);
            mResources.insert(std::make_pair(object->uid, std::static_pointer_cast<TResourceGl>(res)));
            return std::make_pair(std::static_pointer_cast<TResourceGl>(res), true);
        }

        /**
         * @brief Gets the wrapped resource for a certain UID object.
         * @tparam TResourceGl Wrapped type of the resource.
         * @param object Resource with unique ID.
         * @return Pointer to the wrapped resource and a flag that says whether a new wrapper was allocated.
         */
        template <typename TResourceGl>
        [[nodiscard]] std::pair<std::shared_ptr<TResourceGl>, bool> getResource(std::shared_ptr<const Resource> object, OpenGL* opengl)
        {
            // try to find the resource
            auto it = mResources.find(object->uid);

            // does it already exist? return it!
            if (it != mResources.end())
                return std::make_pair(std::static_pointer_cast<TResourceGl>(it->second), false);

            // if it does not exist, then allocate it.
            auto res = allocateResource(object, opengl);
            mResources.insert(make_pair(object->uid, res));
            return std::make_pair(std::static_pointer_cast<TResourceGl>(res), true);
        }

        /**
         * @brief Gets the geometries in the order in which they are in the scene.
         */
        [[nodiscard]] const std::vector<ActorGl*>& linearGeometries() const;

        /**
         * @brief Gets the lights in the order in which they are in the scene.
         */
        [[nodiscard]] const std::vector<ActorGl*>& linearLights() const;

    private:
        /**
         * @brief Allocates a resource for a given object. This is the main wrapper factory.
         * @param object Object to create a wrapper for.
         * @param renderer Renderer that the resource is built for.
         * @param opengl Reference to the openGL handle.
         * @return Wrapped object.
         */
        [[nodiscard]] std::shared_ptr<ActorGl> allocateActor(std::shared_ptr<const Resource> object, const RendererGl* renderer, OpenGL* opengl);

        /**
         * @brief Allocates a resource for a given object. This is the main wrapper factory.
         * @param object Object to create a wrapper for.
         * @param opengl Reference to the openGL handle.
         * @return Wrapped object.
         */
        [[nodiscard]] std::shared_ptr<ResourceBaseGl> allocateResource(std::shared_ptr<const Resource> object, OpenGL* opengl);

        /**
         * @brief Collection of components and resources.
         */
        std::unordered_map<std::size_t, std::shared_ptr<ResourceBaseGl>> mResources;

        /**
         * @brief Geometries in the order in which they are in the scene.
         */
        std::vector<ActorGl*> mLinearGeometries;

        /**
         * @brief Lights in the order in which they are in the scene.
         */
        std::vector<ActorGl*> mLinearLights;
    };
}
