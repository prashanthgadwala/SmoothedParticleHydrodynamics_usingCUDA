#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <string>
#include <vector>

namespace vislab
{
    /**
     * @brief This class is responisble for the initialization of OpenGL.
     */
    class OpenGL
    {
    public:
        /**
         * @brief Struct that represents the viewport parameters.
         */
        struct Viewport
        {
            float width;
            float height;
        };

        /**
         * @brief Base constructor.
         * @param WindowHandle Window handle of GLFW.
         */
        explicit OpenGL(GLFWwindow* WindowHandle);

        /**
         * @brief Destructor.
         */
        virtual ~OpenGL();

        /**
         * @brief Initializes OpenGL by creating the device and the swap chain.
         * @return True if the initialization succeeded.
         */
        [[nodiscard]] bool init();

        /**
         * @brief Gets the viewport description.
         * @return Viewport description.
         */
        [[nodiscard]] inline const Viewport& getViewport() const { return _Viewport; }

        /**
         * @brief Gets the GLFW window handle.
         * @return GLFW window handle.
         */
        [[nodiscard]] inline GLFWwindow* getWindowHandle() { return _WindowHandle; }

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

    private:
        /**
         * @brief Parent Window.
         */
        GLFWwindow* _WindowHandle;

        /**
         * @brief Viewport description.
         */
        Viewport _Viewport;
    };
}
