#include <vislab/opengl/transform_gl.hpp>

#include <vislab/graphics/transform.hpp>

namespace vislab
{
    TransformGl::TransformGl(std::shared_ptr<const Transform> transform)
        : Concrete<TransformGl, ResourceGl<Transform>>(transform)
    {
    }

    TransformGl::~TransformGl()
    {
        this->releaseDevice();
    }

    void TransformGl::update(SceneGl* scene, OpenGL* opengl)
    {
        mParams.data.worldMatrix                    = get()->getMatrix().cast<float>();
        mParams.data.invWorldMatrix                 = mParams.data.worldMatrix.inverse();
        mParams.data.normalMatrix                   = Eigen::Matrix4f::Identity();
        mParams.data.normalMatrix.block(0, 0, 3, 3) = get()->getMatrixInverse().block(0, 0, 3, 3).transpose().cast<float>();
        mParams.updateBuffer();
    }

    bool TransformGl::createDevice(OpenGL* opengl)
    {
        return mParams.createDevice();
    }

    void TransformGl::releaseDevice()
    {
        mParams.releaseDevice();
    }

    bool TransformGl::bind(unsigned int shaderProgram, int bindingPoint)
    {
        return mParams.bind(shaderProgram, "cbtransform", bindingPoint);
    }

    std::string TransformGl::getShaderCode() const noexcept
    {
        return "layout(std140) uniform cbtransform\n"
               "{\n"
               "   mat4 worldMatrix;\n"
               "   mat4 invWorldMatrix;\n"
               "   mat4 normalMatrix;\n"
               "};\n"
               "vec4 transformLocalToWorld(vec4 localPosition) { return worldMatrix * localPosition; }\n"
               "vec4 transformWorldToLocal(vec4 worldPosition) { return invWorldMatrix * worldPosition; }\n"
               "vec3 transformNormalLocalToWorld(vec3 localNormal) { return (normalMatrix * vec4(localNormal, 0.0)).xyz; }\n";
    }
}
