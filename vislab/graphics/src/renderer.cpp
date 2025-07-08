#include <vislab/graphics/renderer.hpp>

#include <thread>

namespace vislab
{
    void Renderer::render()
    {
        ProgressInfo info;
        this->render(info);
    }

    std::future<void> Renderer::renderAsync(ProgressInfo& progress)
    {
        return std::async(
            static_cast<void (Renderer::*)(ProgressInfo&)>(&Renderer::render),
            this,
            std::ref(progress));
    }
}
