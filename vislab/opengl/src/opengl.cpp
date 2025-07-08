#include <vislab/opengl/opengl.hpp>

#include <iostream>

namespace vislab
{
    OpenGL::OpenGL(GLFWwindow* WindowHandle)
        : _WindowHandle(WindowHandle)
        , _Viewport(Viewport())
    {
    }

    OpenGL::~OpenGL()
    {
    }

    bool OpenGL::init()
    {
        if (!createDevice())
            return false;
        if (!createSwapChain())
            return false;
        return true;
    }

    bool OpenGL::createDevice()
    {
        return true;
    }

    bool OpenGL::createSwapChain()
    {
        int width, height;
        glfwGetWindowSize(_WindowHandle, &width, &height);
        _Viewport.width  = (float)width;
        _Viewport.height = (float)height;
        return true;
    }

    void OpenGL::releaseDevice()
    {
    }

    void OpenGL::releaseSwapChain()
    {
    }
}
