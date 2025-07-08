#include <vislab/opengl/camera_gl.hpp>

namespace vislab
{
    CameraGl::CameraGl(std::shared_ptr<const Camera> camera, const std::string& shaderCode)
        : Interface<CameraGl, ResourceGl<Camera>>(camera)
        , mShaderCode(shaderCode)
    {
    }

    const std::string& CameraGl::getShaderCode() const
    {
        return mShaderCode;
    }
}
