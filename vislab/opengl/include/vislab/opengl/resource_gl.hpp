#pragma once

#include <vislab/opengl/resource_base_gl.hpp>

namespace vislab
{
    class SceneGl;

    /**
     * @brief Base class for wrappers of resources.
     * @tparam TResource Type of resource to be wrapped.
     */
    template <typename TResource>
    class ResourceGl : public Interface<ResourceGl<TResource>, ResourceBaseGl>
    {
    public:
        /**
         * @brief Constructor.
         * @param resource Resource to be wrapped.
         */
        ResourceGl(std::shared_ptr<const TResource> resource)
            : mWeakResource(resource)
            , mResource(resource.get())
        {
        }

        /**
         * @brief Updates the resource.
         * @param scene Scene that contains the wrapped resources.
         * @param opengl Reference to the openGL handle.
         */
        virtual void update(SceneGl* scene, OpenGL* opengl) {}

        /**
         * @brief Flag that determines whether the underlying resources was deleted.
         * @return True if the underlying resource was deleted.
         */
        [[nodiscard]] inline bool expired() const
        {
            return mWeakResource.expired();
        }

        /**
         * @brief Gets access to the underlying resource that is wrapped.
         * @return Pointer to the underlying resource.
         */
        [[nodiscard]] inline const TResource* get() const
        {
            return mResource;
        }

    private:
        /**
         * @brief Direct pointer access to underlying resource.
         */
        const TResource* mResource;

        /**
         * @brief Weak pointer reference to the shared pointer that was provided when constructing this wrapper. The weak pointer is used to keep track of the object to be wrapped going out of scope.
         */
        std::weak_ptr<const TResource> mWeakResource;
    };
}
