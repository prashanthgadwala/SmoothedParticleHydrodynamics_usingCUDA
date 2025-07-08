#include "physsim_window.hpp"

#include "imgui_helper.hpp"
#include "simulation.hpp"

#include <vislab/graphics/actor.hpp>
#include <vislab/graphics/bmp_writer.hpp>
#include <vislab/graphics/component.hpp>
#include <vislab/graphics/image.hpp>
#include <vislab/graphics/projective_camera.hpp>
#include <vislab/graphics/scene.hpp>
#include <vislab/graphics/trackball_interactor.hpp>
#include <vislab/opengl/forward_renderer_gl.hpp>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>

namespace physsim
{
    PhyssimWindow::PhyssimWindow(int width, int height, const char* title, std::shared_ptr<Simulation> simulation, bool fullScreen)
        : RenderWindowGl(width, height, title, fullScreen)
        , mSimulation(simulation)
        , mScene(std::make_shared<vislab::Scene>())
        , mForwardRenderer(std::make_shared<vislab::ForwardRendererGl>())
        , mProjectiveCamera(nullptr)
        , mActive(true)
        , mTimeStep(0)
        , mSamplesPerPixel(4)
        , mRecursionDepth(1)
        , mNumFrames(10)
        , mNthTimeStep(10)
    {
    }

    bool PhyssimWindow::init()
    {
        // initialize the simulation and the renderer
        mSimulation->scene = mScene;
        mSimulation->init();
        mSimulation->restart();

        mForwardRenderer->openGL = this->getOpenGL();
        mForwardRenderer->scene = mScene;

        // get the first camera of the scene
        mProjectiveCamera = nullptr;
        for (auto& actor : mScene->actors)
        {
            mProjectiveCamera = actor->components.get<vislab::ProjectiveCamera>();
            if (mProjectiveCamera)
                break;
        }

        mForwardRenderer->camera = mProjectiveCamera;

        // connect the interactor with the camera of the scene
        interactor = std::make_shared<vislab::TrackballInteractor>();
        interactor->setCamera(mProjectiveCamera);

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        // Setup Platform/Renderer bindings
        ImGui_ImplGlfw_InitForOpenGL(getGlwfWindow(), true);
        ImGui_ImplOpenGL3_Init("#version 330");
        // Setup Dear ImGui style
        ImGui::StyleColorsDark();

        return true;
    }

    void PhyssimWindow::restart()
    {
        mTimeStep = 0;
        mSimulation->restart();
    }

    void PhyssimWindow::advance(double elapsedTime, double totalTime)
    {
        mSimulation->advance(elapsedTime, totalTime, mTimeStep);
        mTimeStep++;
    }

    bool PhyssimWindow::createDevice(vislab::OpenGL* opengl)
    {
        return mForwardRenderer->createDevice();
    }

    bool PhyssimWindow::createSwapChain(vislab::OpenGL* opengl)
    {
        return mForwardRenderer->createSwapChain();
    }

    void PhyssimWindow::releaseDevice()
    {
        mForwardRenderer->releaseDevice();
    }

    void PhyssimWindow::releaseSwapChain()
    {
        mForwardRenderer->releaseSwapChain();
    }

    void PhyssimWindow::draw(vislab::OpenGL* opengl, double elapsedTime, double totalTime)
    {
        // advance the simulation one step toward
        if (mActive)
        {
            advance(elapsedTime, totalTime);
        }

        // render the frame with openGL
        // mRenderer->draw(opengl);

        // set camera viewport
        mProjectiveCamera->setWidth(opengl->getViewport().width);
        mProjectiveCamera->setHeight(opengl->getViewport().height);

        mForwardRenderer->update();
        mForwardRenderer->render(mProgress);
        
        // the frame is finished. remove all component changed tags
        for (auto& actor : mScene->actors)
        {
            actor->components.removeTag<vislab::ComponentChangedTag>();
        }

        // draw UI
        gui();
    }

    void PhyssimWindow::gui()
    {
        // if ImGui wants to process mouse events, disable the interactor
        interactor->active = !ImGui::GetIO().WantCaptureMouse;

        // feed inputs to dear imgui, start new frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // draw UI elements
        ImGui::Begin("PhysSim", 0, ImGuiWindowFlags_AlwaysAutoResize);
        ImguiHelper::toggleButton("active", &mActive);
        ImGui::SameLine();
        if (ImGui::Button("Restart"))
            restart();
        ImGui::Text("time step: %ld", mTimeStep);
        ImGui::Separator();

        // simulation UI
        mSimulation->gui();
        ImGui::Separator();

        // close button
        if (ImGui::Button("Close Window"))
            glfwSetWindowShouldClose(getGlwfWindow(), GL_TRUE);

        ImGui::End();

        // render dear imgui onto screen
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
}
