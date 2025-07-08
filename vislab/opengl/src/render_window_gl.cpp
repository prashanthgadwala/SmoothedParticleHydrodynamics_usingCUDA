#include <vislab/opengl/render_window_gl.hpp>

#include <vislab/core/timer.hpp>
#include <vislab/graphics/iinteractor.hpp>
#include <vislab/graphics/key_state.hpp>
#include <vislab/graphics/projective_camera.hpp>
#include <vislab/opengl/opengl.hpp>

#include <iostream>

namespace vislab
{
    RenderWindowGl::RenderWindowGl(int width, int height, const char* title, bool fullScreen)
        : mWnd(NULL)
        , interactor(nullptr)
    {
        if (!initWindow(width, height, title, fullScreen))
            throw std::runtime_error("Window creation failed.");
        mOpenGL = std::make_shared<OpenGL>(mWnd);

        mMouseState.leftDown     = false;
        mMouseState.leftIsDown   = false;
        mMouseState.leftUp       = false;
        mMouseState.rightDown    = false;
        mMouseState.rightIsDown  = false;
        mMouseState.rightUp      = false;
        mMouseState.middleDown   = false;
        mMouseState.middleIsDown = false;
        mMouseState.middleUp     = false;
        mMouseState.shiftDown    = false;
        mMouseState.ctrlDown     = false;
        mMouseState.x            = 0;
        mMouseState.y            = 0;
        mMouseState.width        = 0;
        mMouseState.height       = 0;
        mMouseState.scrollDelta  = 0;
    }

    RenderWindowGl::~RenderWindowGl()
    {
        releaseDevice();
        releaseSwapChain();
        mOpenGL->releaseDevice();
        mOpenGL->releaseSwapChain();
    }

    std::shared_ptr<OpenGL> RenderWindowGl::getOpenGL() { return mOpenGL; }

    bool RenderWindowGl::init()
    {
        return true;
    }

    bool RenderWindowGl::createDevice(OpenGL* opengl)
    {
        return true;
    }

    bool RenderWindowGl::createSwapChain(OpenGL* opengl)
    {
        return true;
    }

    void RenderWindowGl::releaseDevice()
    {
    }

    void RenderWindowGl::releaseSwapChain()
    {
    }

    int RenderWindowGl::run()
    {
        // Initialization of the scene (non-GL resources).
        if (!init())
            return -1;

        // Initialize the GL resource.
        if (!mOpenGL->init())
            return -1;
        if (!createDevice(mOpenGL.get()))
            return -1;
        if (!createSwapChain(mOpenGL.get()))
            return -1;

        // Initialize the timers.
        Timer totalTimer;
        Timer elapsedTimer;
        totalTimer.tic();
        elapsedTimer.tic();

        while (glfwWindowShouldClose(mWnd) == 0)
        {
            mMouseState.scrollDelta = 0;
            mMouseState.shiftDown   = glfwGetKey(mWnd, GLFW_KEY_LEFT_SHIFT) || glfwGetKey(mWnd, GLFW_KEY_RIGHT_SHIFT);
            mMouseState.ctrlDown    = glfwGetKey(mWnd, GLFW_KEY_LEFT_CONTROL) || glfwGetKey(mWnd, GLFW_KEY_RIGHT_CONTROL);
            mMouseState.leftDown    = false;
            mMouseState.leftUp      = false;
            mMouseState.middleDown  = false;
            mMouseState.middleUp    = false;
            mMouseState.rightDown   = false;
            mMouseState.rightUp     = false;

            // process user events
            glfwPollEvents();

            // get elapsed time
            double elapsedTime = elapsedTimer.toc();
            double totalTime   = totalTimer.toc();
            elapsedTimer.tic();

            // draw the frame
            draw(mOpenGL.get(), elapsedTime, totalTime);

            // Swap the front and back buffer.
            glfwSwapBuffers(mWnd);
        }

        glfwTerminate();
        return 0;
    }

    void RenderWindowGl::resize(int width, int height)
    {
        if (mOpenGL == nullptr)
            return;
        releaseSwapChain();
        mOpenGL->releaseSwapChain();
        if (!mOpenGL->createSwapChain())
            return;
        if (!createSwapChain(mOpenGL.get()))
            return;
        glViewport(0, 0, width, height);
    }

    bool RenderWindowGl::initWindow(int width, int height, const char* title, bool fullScreen)
    {
        if (glfwInit() == 0)
            return false;

        // select opengl version
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

        if (fullScreen)
        {
            mWnd = glfwCreateWindow(
                glfwGetVideoMode(glfwGetPrimaryMonitor())->width,
                glfwGetVideoMode(glfwGetPrimaryMonitor())->height,
                title,
                glfwGetPrimaryMonitor(),
                nullptr);
        }
        else
        {
            mWnd = glfwCreateWindow(width, height, title, nullptr, nullptr);
        }

        // check if window was created successfully
        if (mWnd == nullptr)
        {
            return false;
        }

        glfwMakeContextCurrent(mWnd);
        glfwSetWindowUserPointer(mWnd, this);

        // KeyCallback
        auto key_callback = [](GLFWwindow* w, int key, int scancode, int action, int mods)
        {
            static_cast<RenderWindowGl*>(glfwGetWindowUserPointer(w))->keyCallback(w, key, scancode, action, mods);
        };
        glfwSetKeyCallback(mWnd, key_callback);

        // ScrollCallback
        auto scroll_callback = [](GLFWwindow* w, double xoffset, double yoffset)
        {
            static_cast<RenderWindowGl*>(glfwGetWindowUserPointer(w))->scrollCallback(w, xoffset, yoffset);
        };
        glfwSetScrollCallback(mWnd, scroll_callback);

        // CursorPosCallback
        auto cursor_position_callback = [](GLFWwindow* w, double xpos, double ypos)
        {
            static_cast<RenderWindowGl*>(glfwGetWindowUserPointer(w))->cursorPositionCallback(w, xpos, ypos);
        };
        glfwSetCursorPosCallback(mWnd, cursor_position_callback);

        // MouseButtonCallback
        auto mouse_button_callback = [](GLFWwindow* w, int button, int action, int mods)
        {
            static_cast<RenderWindowGl*>(glfwGetWindowUserPointer(w))->mouseButtonCallback(w, button, action, mods);
        };
        glfwSetMouseButtonCallback(mWnd, mouse_button_callback);

        // WindowSizeCallback
        auto window_size_callback = [](GLFWwindow* w, int width, int height)
        {
            static_cast<RenderWindowGl*>(glfwGetWindowUserPointer(w))->windowSizeCallback(w, width, height);
        };
        glfwSetWindowSizeCallback(mWnd, window_size_callback);

        // initialize glew
        if (glewInit() != GLEW_OK)
        {
            glfwDestroyWindow(mWnd);
            glfwTerminate();
            return false;
        }
        return true;
    }

    void RenderWindowGl::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        KeyState mKeyState     = {};
        mKeyState.isDown_A     = key == GLFW_KEY_A && action == GLFW_PRESS;
        mKeyState.isDown_B     = key == GLFW_KEY_B && action == GLFW_PRESS;
        mKeyState.isDown_C     = key == GLFW_KEY_C && action == GLFW_PRESS;
        mKeyState.isDown_D     = key == GLFW_KEY_D && action == GLFW_PRESS;
        mKeyState.isDown_E     = key == GLFW_KEY_E && action == GLFW_PRESS;
        mKeyState.isDown_F     = key == GLFW_KEY_F && action == GLFW_PRESS;
        mKeyState.isDown_G     = key == GLFW_KEY_G && action == GLFW_PRESS;
        mKeyState.isDown_H     = key == GLFW_KEY_H && action == GLFW_PRESS;
        mKeyState.isDown_I     = key == GLFW_KEY_I && action == GLFW_PRESS;
        mKeyState.isDown_J     = key == GLFW_KEY_J && action == GLFW_PRESS;
        mKeyState.isDown_K     = key == GLFW_KEY_K && action == GLFW_PRESS;
        mKeyState.isDown_L     = key == GLFW_KEY_L && action == GLFW_PRESS;
        mKeyState.isDown_M     = key == GLFW_KEY_M && action == GLFW_PRESS;
        mKeyState.isDown_N     = key == GLFW_KEY_N && action == GLFW_PRESS;
        mKeyState.isDown_O     = key == GLFW_KEY_O && action == GLFW_PRESS;
        mKeyState.isDown_P     = key == GLFW_KEY_P && action == GLFW_PRESS;
        mKeyState.isDown_Q     = key == GLFW_KEY_Q && action == GLFW_PRESS;
        mKeyState.isDown_R     = key == GLFW_KEY_R && action == GLFW_PRESS;
        mKeyState.isDown_S     = key == GLFW_KEY_S && action == GLFW_PRESS;
        mKeyState.isDown_T     = key == GLFW_KEY_T && action == GLFW_PRESS;
        mKeyState.isDown_U     = key == GLFW_KEY_U && action == GLFW_PRESS;
        mKeyState.isDown_V     = key == GLFW_KEY_V && action == GLFW_PRESS;
        mKeyState.isDown_W     = key == GLFW_KEY_W && action == GLFW_PRESS;
        mKeyState.isDown_X     = key == GLFW_KEY_X && action == GLFW_PRESS;
        mKeyState.isDown_Y     = key == GLFW_KEY_Y && action == GLFW_PRESS;
        mKeyState.isDown_Z     = key == GLFW_KEY_Z && action == GLFW_PRESS;
        mKeyState.isDown_Space = key == GLFW_KEY_SPACE && action == GLFW_PRESS;
        if (interactor && interactor->active)
            interactor->onKeyEvent(mKeyState);
    }

    void RenderWindowGl::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
    {
        mMouseState.scrollDelta = yoffset;
        if (interactor && interactor->active)
            interactor->onMouseEvent(mMouseState);
    }

    void RenderWindowGl::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
    {
        mMouseState.x = xpos;
        mMouseState.y = ypos;
        if (interactor && interactor->active)
            interactor->onMouseEvent(mMouseState);
    }

    void RenderWindowGl::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
    {
        if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if (action == GLFW_PRESS)
            {
                mMouseState.rightUp     = false;
                mMouseState.rightIsDown = true;
                mMouseState.rightDown   = true;
            }
            else if (action == GLFW_RELEASE)
            {
                mMouseState.rightUp     = true;
                mMouseState.rightIsDown = false;
                mMouseState.rightDown   = false;
            }
        }
        else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
        {
            if (action == GLFW_PRESS)
            {
                mMouseState.middleUp     = false;
                mMouseState.middleIsDown = true;
                mMouseState.middleDown   = true;
            }
            else if (action == GLFW_RELEASE)
            {
                mMouseState.middleUp     = true;
                mMouseState.middleIsDown = false;
                mMouseState.middleDown   = false;
            }
        }
        else if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            if (action == GLFW_PRESS)
            {
                mMouseState.leftUp     = false;
                mMouseState.leftIsDown = true;
                mMouseState.leftDown   = true;
            }
            else if (action == GLFW_RELEASE)
            {
                mMouseState.leftUp     = true;
                mMouseState.leftIsDown = false;
                mMouseState.leftDown   = false;
            }
        }
        if (interactor && interactor->active)
            interactor->onMouseEvent(mMouseState);
    }

    void RenderWindowGl::windowSizeCallback(GLFWwindow* window, int width, int height)
    {
        resize(width, height);
    }
}
