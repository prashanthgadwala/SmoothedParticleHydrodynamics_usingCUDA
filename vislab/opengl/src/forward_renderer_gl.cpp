#include <vislab/opengl/forward_renderer_gl.hpp>

#include <vislab/core/array.hpp>
#include <vislab/field/regular_field.hpp>
#include <vislab/graphics/actor.hpp>
#include <vislab/graphics/colormap_texture.hpp>
#include <vislab/graphics/const_texture.hpp>
#include <vislab/graphics/diffuse_bsdf.hpp>
#include <vislab/graphics/point_light.hpp>
#include <vislab/graphics/projective_camera.hpp>
#include <vislab/graphics/rectangle_geometry.hpp>
#include <vislab/graphics/scene.hpp>
#include <vislab/graphics/sphere_geometry.hpp>
#include <vislab/graphics/transform.hpp>
#include <vislab/graphics/trimesh_geometry.hpp>
#include <vislab/opengl/actor_gl.hpp>
#include <vislab/opengl/bsdf_gl.hpp>
#include <vislab/opengl/geometry_gl.hpp>
#include <vislab/opengl/opengl.hpp>
#include <vislab/opengl/point_light_gl.hpp>
#include <vislab/opengl/projective_camera_gl.hpp>
#include <vislab/opengl/scene_gl.hpp>
#include <vislab/opengl/shader_gl.hpp>
#include <vislab/opengl/transform_gl.hpp>

namespace vislab
{
    ForwardRendererGl::ForwardRendererGl()
        : Concrete<ForwardRendererGl, RendererGl>(generateShaderCodes())
    {
    }

    ForwardRendererGl::~ForwardRendererGl()
    {
        releaseDevice();
        releaseSwapChain();
    }

    void ForwardRendererGl::render()
    {
        ProgressInfo info;
        this->render(info);
    }

    void ForwardRendererGl::render(ProgressInfo& progressInfo)
    {
        // get the camera resource
        std::shared_ptr<ProjectiveCameraGl> projectiveCamera = mScene->getResource<ProjectiveCameraGl>(this->camera, openGL.get()).first;
        if (!projectiveCamera)
            return;

        // clear the backbuffer and depth buffer
        Eigen::Vector4f clearColor(1, 1, 1, 1);
        glClearColor(clearColor.x(), clearColor.y(), clearColor.z(), clearColor.w());
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_FRAMEBUFFER_SRGB);

        // setup depth writing and test
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);

        // compile light list
        auto& lightActors = mScene->linearLights();
        int ilight        = 0;
        for (auto& lightActor : lightActors)
        {
            auto pointLight = std::dynamic_pointer_cast<PointLightGl>(lightActors[ilight]->light);
            if (!pointLight)
                continue;
            Spectrum lightIntensity                  = pointLight->get()->intensity;
            Eigen::Vector3d lightPos                 = lightActors[ilight]->transform->get()->getMatrix().translation();
            mLightParams.data.lightPos[ilight]       = Eigen::Vector4f(lightPos.x(), lightPos.y(), lightPos.z(), 1);
            mLightParams.data.lightIntensity[ilight] = Eigen::Vector4f(lightIntensity.x(), lightIntensity.y(), lightIntensity.z(), 1);
            ilight++;
            if (ilight == 8)
                break;
        }
        for (int i = ilight; i < 8; ++i)
        {
            mLightParams.data.lightIntensity[i] = Eigen::Vector4f::Zero();
            mLightParams.data.lightPos[i]       = Eigen::Vector4f::Zero();
        }
        mLightParams.updateBuffer();

        // traverse the geometries for rendering
        auto& geometryActors = mScene->linearGeometries();
        for (auto& geometryActor : geometryActors)
        {
            // bind shader
            std::size_t pass = geometryActor->geometry->writesDepth ? 1 : 0;
            auto shader = geometryActor->shaders.find(uid + pass);
            if (shader == geometryActor->shaders.end())
                continue;
            shader->second->bind();
            unsigned int shaderProgram = shader->second->getProgram();

            // bind camera
            if (!projectiveCamera->bind(shaderProgram, 0))
                continue;

            // bind lights
            if (!mLightParams.bind(shaderProgram, "cbrender", 1))
                continue;

            // bind transform params
            if (!geometryActor->transform->bind(shaderProgram, 2))
                continue;

            // bind bsdf params
            if (!geometryActor->bsdf->bind(shaderProgram, 0, 0, 1, 3, 4))
                continue;

            // submit draw call on geometry
            if (!geometryActor->geometry->bind(shaderProgram, 5))
                continue;
            geometryActor->geometry->draw();

            // unbind
            glUseProgram(0);
        }

        progressInfo.allJobsDone();

        int test = 0;
        ++test;
    }

    bool ForwardRendererGl::createDevice()
    {
        if (!openGL)
            return false;

        if (!mLightParams.createDevice())
            return false;

        return true;
    }

    bool ForwardRendererGl::createSwapChain()
    {
        return true;
    }

    void ForwardRendererGl::releaseDevice()
    {
        mLightParams.releaseDevice();
    }

    void ForwardRendererGl::releaseSwapChain()
    {
    }

    std::vector<ShaderSourceGl> ForwardRendererGl::generateShaderCodes()
    {
        ShaderSourceGl defaultSource;
        defaultSource.vertexShader =
            "void main()\n"
            "{\n"
            "   vs_geometry_default();"
            "}\n";

        defaultSource.geometryShader =
            "void main()\n"
            "{\n"
            "   gs_geometry_default();"
            "}\n";
        defaultSource.pixelShader =
            "layout(std140) uniform cbrender\n"
            "{\n"
            "	vec4 lightPos[8];\n"
            "	vec4 lightIntensity[8];\n"
            "};\n" +
            PointLightGl::generateCode() +
            "out vec4 ps_out_FragColor;\n"
            "void main()\n"
            "{\n"
            "   vec3 p, n;\n"
            "   vec2 uv;\n"
            "   float data, depth;\n"
            "   ps_geometry_default(p, n, uv, data, depth);\n"
            "   vec3 color = vec3(0,0,0);\n"
            "   for (int i = 0; i < 8; ++i) {\n"
            "      vec3 wo;\n"
            "      vec3 Li = sample_pointlight(p, lightPos[i].xyz, lightIntensity[i].rgb, wo);\n"
            "      vec3 wi = normalize(eye() - p);\n"
            "      color += evaluate_bsdf(wi, wo, n, uv, data) * Li;\n"
            "   }\n"
            "   ps_out_FragColor = vec4(color, 1);\n"
            "}\n";

        ShaderSourceGl writeDepth;
        writeDepth.vertexShader   = defaultSource.vertexShader;
        writeDepth.geometryShader = defaultSource.geometryShader;
        writeDepth.pixelShader =
            "layout(std140) uniform cbrender\n"
            "{\n"
            "	vec4 lightPos[8];\n"
            "	vec4 lightIntensity[8];\n"
            "};\n" +
            PointLightGl::generateCode() +
            "out vec4 ps_out_FragColor;\n"
            "void main()\n"
            "{\n"
            "   vec3 p, n;\n"
            "   vec2 uv;\n"
            "   float data, depth;\n"
            "   ps_geometry_default(p, n, uv, data, depth);\n"
            "   vec3 color = vec3(0,0,0);\n"
            "   for (int i = 0; i < 8; ++i) {\n"
            "      vec3 wo;\n"
            "      vec3 Li = sample_pointlight(p, lightPos[i].xyz, lightIntensity[i].rgb, wo);\n"
            "      vec3 wi = normalize(eye() - p);\n"
            "      color += evaluate_bsdf(wi, wo, n, uv, data) * Li;\n"
            "   }\n"
            "   ps_out_FragColor = vec4(color, 1);\n"
            "   gl_FragDepth = depth;\n" // <-- write depth
            "}\n";
        ;

        return std::vector<ShaderSourceGl>({ defaultSource, writeDepth });
    };
}
