#include <vislab/opengl/renderer_gl.hpp>

#include <vislab/opengl/scene_gl.hpp>

namespace vislab
{
    static std::size_t generateUniqueIdentifier(std::size_t increment)
    {
        static std::size_t global_uid = 0;
        std::size_t current_uid       = global_uid;
        global_uid += increment;
        return current_uid;
    }

    RendererGl::RendererGl(const std::vector<ShaderSourceGl>& shaderCodes)
        : passes(shaderCodes)
        , uid(generateUniqueIdentifier(shaderCodes.size()))
        , mScene(std::make_shared<SceneGl>())
    {
    }

    RendererGl::~RendererGl()
    {
    }

    void RendererGl::update()
    {
        mScene->update(openGL.get(), scene.get(), this);
    }
}
