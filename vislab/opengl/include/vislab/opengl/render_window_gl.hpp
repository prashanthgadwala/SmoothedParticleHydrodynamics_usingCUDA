#pragma once

#include <vislab/graphics/mouse_state.hpp>

#include "Eigen/Eigen"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <memory>

namespace vislab
{
    class OpenGL;
    class IInteractor;

    /**
     * @brief Class that creates and handles a window with OpenGL context.
     */
    class RenderWindowGl
    {
    public:
        /**
         * @brief Constructor.
         * @param width Width of the window (ignored when fullScreen is on).
         * @param height Height of the window (ignored when fullScreen is on).
         * @param title Title of the window.
         * @param fullScreen Flag that turns full screen rendering on.
         */
        RenderWindowGl(int width, int height, const char* title, bool fullScreen = false);

        /**
         * @brief Destructor.
         */
        ~RenderWindowGl();

        /**
         * @brief Launches the application.
         * @return Application error code.
         */
        [[nodiscard]] int run();

        /**
         * @brief Comprises context and backbuffer creation.
         * @return OpenGL context.
         */
        [[nodiscard]] std::shared_ptr<OpenGL> getOpenGL();

        /**
         * @brief Optional: Interactor that can control a camera.
         */
        std::shared_ptr<IInteractor> interactor;

    protected:
        /**
         * @brief Initialization before the creation of GL context.
         * @return True, if the creation succeeded.
         */
        [[nodiscard]] virtual bool init();

        /**
         * @brief Creates the device resources.
         * @param opengl Reference to the openGL handle.
         * @return True, if the creation succeeded.
         */
        [[nodiscard]] virtual bool createDevice(OpenGL* opengl);

        /**
         * @brief Creates the swap chain resources, i.e., resources that depend on the screen resolution.
         * @param opengl Reference to the openGL handle.
         * @return True, if the creation succeeded.
         */
        [[nodiscard]] virtual bool createSwapChain(OpenGL* opengl);

        /**
         * @brief Releases the device resources.
         */
         virtual void releaseDevice();

        /**
         * @brief Releases the swap chain resources.
         */
         virtual void releaseSwapChain();

        /**
         * @brief Draws the frame by invoking OpenGL calls.
         * @param opengl Handle to the openGL context.
         * @param elapsedTime elapsed time in milliseconds during the last frame.
         * @param totalTime total time in milliseconds since the beginning of the current frame.
         */
        virtual void draw(OpenGL* opengl, double elapsedTime, double totalTime) = 0;

        /**
         * @brief Resizes the swap chain
         * @param width New width of the viewport.
         * @param height New height of the viewport.
         */
        void resize(int width, int height);

    private:
        /**
         * @brief Initializes a window.
         * @param width Width of the window (ignored when fullScreen is on).
         * @param height Height of the window (ignored when fullScreen is on).
         * @param title Title of the window.
         * @param fullScreen Flag that turns full screen rendering on.
         * @return True if successful.
         */
        [[nodiscard]] bool initWindow(int width, int height, const char* title, bool fullScreen);

        /**
         * @brief Callback that handles key strokes.
         * @param window GLFW window handle.
         * @param key Key that was pressed.
         * @param scancode Scan code.
         * @param action Action.
         * @param mods Modifiers.
         */
        virtual void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

        /**
         * @brief Callback that handles mouse scoll events.
         * @param window GLFW window handle.
         * @param xoffset Horizontal scroll amount.
         * @param yoffset Vertical scroll amount.
         */
        virtual void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

        /**
         * @brief Callback that handles mouse move events.
         * @param window GLFW window handle.
         * @param xpos Horizontal position of the cursor.
         * @param ypos Vertical position of the cursor.
         */
        virtual void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);

        /**
         * @brief Callback that handles mouse button press events.
         * @param window GLFW window handle.
         * @param button Button that was pressed.
         * @param action Action.
         * @param mods Modifiers.
         */
        virtual void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

        /**
         * @brief Callback that handles window resize events.
         * @param window GLFW window handle.
         * @param width New width of window.
         * @param height New height of window.
         */
        virtual void windowSizeCallback(GLFWwindow* window, int width, int height);

    protected:
        /**
         * @brief Gets the GLFW window handle.
         * @return Window handle.
         */
        [[nodiscard]] inline GLFWwindow* getGlwfWindow() { return mWnd; }

    private:
        /**
         * @brief Window handle
         */
        GLFWwindow* mWnd;

        /**
         * @brief Comprises context and backbuffer creation.
         */
        std::shared_ptr<OpenGL> mOpenGL;

        /**
         * @brief Mouse state.
         */
        MouseState mMouseState;
    };
}
