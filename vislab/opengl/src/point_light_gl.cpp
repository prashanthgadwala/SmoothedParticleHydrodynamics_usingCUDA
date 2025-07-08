#include <vislab/opengl/point_light_gl.hpp>

#include <vislab/graphics/point_light.hpp>

namespace vislab
{
    PointLightGl::PointLightGl(std::shared_ptr<const PointLight> pointLight)
        : Concrete<PointLightGl, LightGl>(pointLight)
    {
    }

    PointLightGl::~PointLightGl()
    {
        this->releaseDevice();
    }

    void PointLightGl::update(SceneGl* scene, OpenGL* opengl)
    {
    }

    bool PointLightGl::createDevice(OpenGL* opengl)
    {
        return true;
    }

    void PointLightGl::releaseDevice()
    {
    }

    const PointLight* PointLightGl::get() const
    {
        return static_cast<const PointLight*>(LightGl::get());
    }

    std::string PointLightGl::generateCode() noexcept
    {
        return
            // p = position of sample point on light receiver
            // l = position of point light
            // I = intensity of point light
            // wo = normalized outgoing direction towards the light source
            "vec3 sample_pointlight(vec3 p, vec3 l, vec3 I, out vec3 wo)\n"
            "{\n"
            "   vec3 lp = l - p;\n"
            "   float dist2 = dot(lp, lp);\n"
            "   wo = normalize(lp);\n"
            "   return I / dist2;\n"
            "}\n";
    }
}
