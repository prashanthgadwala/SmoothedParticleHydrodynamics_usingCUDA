#include <vislab/opengl/projective_camera_gl.hpp>

#include <vislab/graphics/camera.hpp>
#include <vislab/graphics/projective_camera.hpp>

namespace vislab
{
    ProjectiveCameraGl::ProjectiveCameraGl(std::shared_ptr<const ProjectiveCamera> camera)
        : Concrete<ProjectiveCameraGl, CameraGl>(camera, generateShaderCode())
    {
    }

    ProjectiveCameraGl::~ProjectiveCameraGl()
    {
        this->releaseDevice();
    }

    void ProjectiveCameraGl::update(SceneGl* scene, OpenGL* opengl)
    {
        mParams.data.viewMatrix        = get()->getView().cast<float>();
        mParams.data.projMatrix        = get()->getProj().cast<float>();
        mParams.data.eyePosition       = Eigen::Vector4d(get()->getPosition().x(), get()->getPosition().y(), get()->getPosition().z(), 1.).cast<float>();
        mParams.data.invViewProjMatrix = (mParams.data.projMatrix * mParams.data.viewMatrix).inverse();
        mParams.updateBuffer();
    }

    bool ProjectiveCameraGl::createDevice(OpenGL* opengl)
    {
        return mParams.createDevice();
    }

    void ProjectiveCameraGl::releaseDevice()
    {
        mParams.releaseDevice();
    }

    bool ProjectiveCameraGl::bind(unsigned int shaderProgram, int bindingPoint)
    {
        return mParams.bind(shaderProgram, "cbcamera", bindingPoint);
    }

    const ProjectiveCamera* ProjectiveCameraGl::get() const
    {
        return static_cast<const ProjectiveCamera*>(CameraGl::get());
    }

    std::string ProjectiveCameraGl::generateShaderCode()
    {
        return "layout(std140) uniform cbcamera\n"
               "{\n"
               "   mat4 viewMatrix;\n"
               "   mat4 projMatrix;\n"
               "   mat4 invViewProjMatrix;\n"
               "   vec4 eyePosition;\n"
               "};\n"
               "vec4 transformWorldToView(vec4 worldPosition) { return viewMatrix * worldPosition; }\n"
               "vec4 transformViewToClip(vec4 viewPosition) { return projMatrix * viewPosition; }\n"
               "vec4 transformWorldToClip(vec4 worldPosition) { return projMatrix * viewMatrix * worldPosition; }\n"
               "vec4 transformClipToWorld(vec4 clipPosition) { return invViewProjMatrix * clipPosition; }\n"
               "vec3 eye() { return eyePosition.xyz; }\n";
    }
}
