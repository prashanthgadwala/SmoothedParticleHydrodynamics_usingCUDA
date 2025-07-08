#pragma once

#include <vislab/core/object.hpp>

namespace vislab
{
    class OpenGL;

    /**
     * @brief Interface for resource wrappers.
     */
    class ResourceBaseGl : public Interface<ResourceBaseGl, Object>
    {
    public:
        /**
         * @brief Flag that determines whether the underlying resources was deleted.
         * @return True if the underlying resource was deleted.
         */
        [[nodiscard]] virtual bool expired() const = 0;

        /**
         * @brief Creates the device resources.
         * @param opengl Reference to the openGL handle.
         * @return True, if the creation succeeded.
         */
        [[nodiscard]] virtual bool createDevice(OpenGL* opengl) = 0;

        /**
         * @brief Releases the device resources.
         */
        virtual void releaseDevice() = 0;
    };
}
