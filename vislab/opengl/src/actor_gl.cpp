#include <vislab/opengl/actor_gl.hpp>

#include <vislab/graphics/actor.hpp>
#include <vislab/graphics/bsdf.hpp>
#include <vislab/graphics/camera.hpp>
#include <vislab/graphics/geometry.hpp>
#include <vislab/graphics/light.hpp>
#include <vislab/graphics/transform.hpp>

#include <vislab/opengl/bsdf_gl.hpp>
#include <vislab/opengl/camera_gl.hpp>
#include <vislab/opengl/geometry_gl.hpp>
#include <vislab/opengl/light_gl.hpp>
#include <vislab/opengl/renderer_gl.hpp>
#include <vislab/opengl/scene_gl.hpp>
#include <vislab/opengl/shader_gl.hpp>
#include <vislab/opengl/transform_gl.hpp>

namespace vislab
{
    ActorGl::ActorGl(std::shared_ptr<const Actor> actor)
        : Concrete<ActorGl, ResourceGl<Actor>>(actor)
    {
    }

    Eigen::AlignedBox3d ActorGl::worldBounds() const
    {
        Eigen::AlignedBox3d oob = geometry->get()->objectBounds();
        if (!transform)
            return oob;
        return transform->get()->transformBox(oob);
    }

    void ActorGl::update(SceneGl* scene, const RendererGl* renderer, OpenGL* opengl)
    {
        const Actor* actor = get();

        // get the components
        auto bsdf_      = actor->components.get<BSDF>();
        auto light_     = actor->components.get<Light>();
        auto geometry_  = actor->components.get<Geometry>();
        auto transform_ = actor->components.get<Transform>();
        auto camera_    = actor->components.get<Camera>();

        // regrab the wrapped components
        bsdf      = bsdf_ ? scene->getResource<BSDFGl>(bsdf_, opengl).first : nullptr;
        light     = light_ ? scene->getResource<LightGl>(light_, opengl).first : nullptr;
        geometry  = geometry_ ? scene->getResource<GeometryGl>(geometry_, opengl).first : nullptr;
        transform = transform_ ? scene->getResource<TransformGl>(transform_, opengl).first : nullptr;
        camera    = camera_ ? scene->getResource<CameraGl>(camera_, opengl).first : nullptr;

        // is renderable actor
        if (geometry && transform && bsdf)
        {
            // shader code of the active camera
            auto activeCamera = renderer->camera ? scene->getResource<CameraGl>(renderer->camera, opengl).first : nullptr;
            if (!activeCamera)
                return;
            std::string cameraShaderCode = activeCamera->getShaderCode();

            // shader code of the renderer
            for (std::size_t ipass = 0; ipass < renderer->passes.size(); ++ipass)
            {
                // assemble the shader for this specific renderer/camera combination.
                assembleShaderCode(renderer->uid + ipass, renderer->passes[ipass], cameraShaderCode);
            }
        }
    }

    void ActorGl::assembleShaderCode(std::size_t uid, const ShaderSourceGl& rendererShaderCode, const std::string& cameraShaderCode)
    {
        if (geometry && transform && bsdf)
        {
            // get base code of the geometry shader
            std::string geometry_vs_code = geometry->sourceCode.vertexShader;
            std::string geometry_gs_code = geometry->sourceCode.geometryShader;
            std::string geometry_ps_code = geometry->sourceCode.pixelShader;

            // get routines from the other components
            std::string transform_code = transform->getShaderCode();
            std::string bsdf_code      = bsdf->getShaderCode();

            // assemble shader code
            std::string vs_code = "", gs_code = "", ps_code = "";
            if (geometry_vs_code != "")
                vs_code = "#version 430 core\n" + transform_code + cameraShaderCode + geometry_vs_code + rendererShaderCode.vertexShader;
            if (geometry_gs_code != "")
                gs_code = "#version 430 core\n" + transform_code + cameraShaderCode + geometry_gs_code + rendererShaderCode.geometryShader;
            if (geometry_ps_code != "")
                ps_code = "#version 430 core\n" + transform_code + cameraShaderCode + bsdf_code + geometry_ps_code + rendererShaderCode.pixelShader;

            // assemble the shader
            auto shader                       = std::make_shared<ShaderGl>();
            shader->sourceCode.vertexShader   = vs_code;
            shader->sourceCode.geometryShader = gs_code;
            shader->sourceCode.pixelShader    = ps_code;
            if (!shader->createDevice())
                return;
            shaders.insert(std::make_pair(uid, shader));
        }
    }

    bool ActorGl::createDevice(OpenGL* opengl)
    {
        return true;
    }

    void ActorGl::releaseDevice()
    {
    }
}
