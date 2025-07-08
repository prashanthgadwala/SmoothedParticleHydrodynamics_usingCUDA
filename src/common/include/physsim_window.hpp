#pragma once

#include <vislab/opengl/render_window_gl.hpp>
#include <vislab/core/progress_info.hpp>

namespace vislab
{
    class Scene;
    class OpenGL;
    class ProjectiveCamera;
    class ForwardRendererGl;
}

namespace physsim
{
    class Simulation;

    /**
     * @brief Implements a window with basic UI for the physics simulation course.
     */
    class PhyssimWindow : public vislab::RenderWindowGl
    {
    public:
        /**
         * @brief Constructor.
         * @param width Width of the window (ignored when fullScreen is on).
         * @param height Height of the window (ignored when fullScreen is on).
         * @param title Title of the window.
         * @param simulation Simulation to carry out.
         * @param fullScreen Flag that turns full screen rendering on.
         */
        PhyssimWindow(int width, int height, const char* title, std::shared_ptr<Simulation> simulation, bool fullScreen = false);

    private:
        /**
         * @brief Initialization before the creation of GL context.
         * @return True, if the creation succeeded.
         */
        bool init() override;

        /**
         * @brief Restarts the simulation.
         */
        void restart();

        /**
         * @brief Advances the simulation one time step forward.
         * @param elapsedTime Elapsed time in milliseconds during the last frame.
         * @param totalTime Total time in milliseconds since the beginning of the first frame.
         * @param timeStep Time step of the simulation. Restarts when resetting the simulation.
         */
        void advance(double elapsedTime, double totalTime);

        /**
         * @brief Creates the device resources.
         * @param opengl Reference to the openGL handle.
         * @return True, if the creation succeeded.
         */
        bool createDevice(vislab::OpenGL* opengl) override;

        /**
         * @brief Creates the swap chain resources, i.e., resources that depend on the screen resolution.
         * @param opengl Reference to the openGL handle.
         * @return True, if the creation succeeded.
         */
        bool createSwapChain(vislab::OpenGL* opengl) override;

        /**
         * @brief Releases the device resources.
         */
        void releaseDevice() override;

        /**
         * @brief Releases the swap chain resources.
         */
        void releaseSwapChain() override;

        /**
         * @brief Draws the frame by invoking OpenGL calls.
         * @param opengl Handle to the openGL context.
         * @param elapsedTime Elapsed time in milliseconds during the last frame.
         * @param totalTime Total time in milliseconds since the beginning of the first frame.
         */
        void draw(vislab::OpenGL* opengl, double elapsedTime, double totalTime) override;

        /**
         * @brief Adds graphical user interface elements.
         */
        void gui();

        /**
         * @brief Renders the current view with a path tracer.
         */
        void pathTrace();

        /**
         * @brief Exports a video.
         */
        void exportVideo();

        /**
         * @brief Stores the content of the scene.
         */
        std::shared_ptr<vislab::Scene> mScene;

        /**
         * @brief Implements the physics simulation.
         */
        std::shared_ptr<Simulation> mSimulation;

        /**
         * @brief Renderer of the scene.
         */
        std::shared_ptr<vislab::ForwardRendererGl> mForwardRenderer;


        /**
         * @brief Projective camera that is used by the physsim window. The window holds a reference to this non-const camera, since it adjusts the width and height of the viewport when the window is resized.
         */
        std::shared_ptr<vislab::ProjectiveCamera> mProjectiveCamera;

        /**
         * @brief Flag that determines whether the simulation is active or paused.
         */
        bool mActive;

        /**
         * @brief Time step of the simulation.
         */
        int64_t mTimeStep;

        /**
         * @brief Number of samples per pixel in the Monte Carlo renderer.
         */
        int mSamplesPerPixel;

        /**
         * @brief Maximal recursion depth in the Monte Carlo renderer.
         */
        int mRecursionDepth;

        /**
         * @brief Number of frames to export for video.
         */
        int mNumFrames;

        /**
         * @brief Export only every Nths time step in the video.
         */
        int mNthTimeStep;

        /**
         * @brief Captures progress for rendering.
         *
         */
        vislab::ProgressInfo mProgress;
    };
}
