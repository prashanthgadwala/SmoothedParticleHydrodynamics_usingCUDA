#include <vislab/opengl/diffuse_bsdf_gl.hpp>

#include <vislab/opengl/colormap_texture_gl.hpp>
#include <vislab/opengl/opengl.hpp>
#include <vislab/opengl/scene_gl.hpp>
#include <vislab/opengl/transfer_function_gl.hpp>

#include <vislab/graphics/colormap_texture.hpp>
#include <vislab/graphics/const_texture.hpp>
#include <vislab/graphics/diffuse_bsdf.hpp>
#include <vislab/graphics/texture.hpp>

namespace vislab
{
    DiffuseBSDFGl::DiffuseBSDFGl(std::shared_ptr<const DiffuseBSDF> bsdf)
        : Concrete<DiffuseBSDFGl, BSDFGl>(bsdf, generateShaderCode())
        , mColormapTexture(nullptr)
    {
    }

    DiffuseBSDFGl::~DiffuseBSDFGl()
    {
        this->releaseDevice();
    }

    void DiffuseBSDFGl::update(SceneGl* scene, OpenGL* opengl)
    {
        auto const_texture_    = std::dynamic_pointer_cast<const vislab::ConstTexture>(get()->reflectance);
        auto colormap_texture_ = std::dynamic_pointer_cast<const vislab::ColormapTexture>(get()->reflectance);

        mColormapTexture = colormap_texture_ ? scene->getResource<ColormapTextureGl>(colormap_texture_, opengl).first : nullptr;

        // set bsdf parameters
        if (const_texture_)
        {
            mParams.data.reflectance = EReflectance::Constant;
            mParams.data.diffuse     = Eigen::Vector4f(
                static_cast<float>(const_texture_->color.x()),
                static_cast<float>(const_texture_->color.y()),
                static_cast<float>(const_texture_->color.z()),
                1.f);
        }
        else if (colormap_texture_)
        {
            mColormapTexture->update(scene, opengl);
            if (colormap_texture_->scalarField)
                mParams.data.reflectance = EReflectance::FromScalarField;
            else
                mParams.data.reflectance = EReflectance::FromData;
            mParams.data.diffuse = Eigen::Vector4f::Ones();
        }
        else
        {
            mParams.data.reflectance = EReflectance::Constant;
            mParams.data.diffuse     = Eigen::Vector4f::Ones();
        }

        mParams.updateBuffer();
    }

    bool DiffuseBSDFGl::createDevice(OpenGL* opengl)
    {
        auto const_texture_    = std::dynamic_pointer_cast<const vislab::ConstTexture>(get()->reflectance);
        auto colormap_texture_ = std::dynamic_pointer_cast<const vislab::ColormapTexture>(get()->reflectance);

        // set bsdf parameters
        if (const_texture_)
        {
            mParams.data.reflectance = EReflectance::Constant;
            mParams.data.diffuse     = Eigen::Vector4f(
                static_cast<float>(const_texture_->color.x()),
                static_cast<float>(const_texture_->color.y()),
                static_cast<float>(const_texture_->color.z()),
                1.f);
        }
        else if (colormap_texture_)
        {
            if (colormap_texture_->scalarField)
                mParams.data.reflectance = EReflectance::FromScalarField;
            else
                mParams.data.reflectance = EReflectance::FromData;
            mParams.data.diffuse = Eigen::Vector4f::Ones();
        }
        else
        {
            mParams.data.reflectance = EReflectance::Constant;
            mParams.data.diffuse     = Eigen::Vector4f::Ones();
        }

        return mParams.createDevice();
    }

    void DiffuseBSDFGl::releaseDevice()
    {
        if (mColormapTexture)
            mColormapTexture->releaseDevice();
        mParams.releaseDevice();
    }

    bool DiffuseBSDFGl::bind(unsigned int shaderProgram, int textureBindingPoint, int keyBindingPoint, int valuesBindingPoint, int cbBindingPointTf, int cbBindingPointBsdf)
    {
        if (mColormapTexture && !mColormapTexture->bind(shaderProgram, textureBindingPoint, keyBindingPoint, valuesBindingPoint, cbBindingPointTf))
            return false;
        return mParams.bind(shaderProgram, "cbbsdf", cbBindingPointBsdf);
    }

    const DiffuseBSDF* DiffuseBSDFGl::get() const
    {
        return static_cast<const DiffuseBSDF*>(BSDFGl::get());
    }

    std::string DiffuseBSDFGl::generateShaderCode() noexcept
    {
        // clang-format off
        return "layout(std140) uniform cbbsdf\n "
               "{\n"
               "   vec4 diffuse;\n"
               "   int reflectanceSource;\n"
               "};\n"
               "uniform sampler2D texColormap;\n"
               + ColormapTextureGl::generateCode() +
               "vec3 evaluate_bsdf(vec3 wi, vec3 wo, vec3 n, vec2 uv, float data)\n"
               "{\n"
               "   vec3 reflectance = diffuse.rgb;\n"
               "   if (reflectanceSource == 1)\n" // from data
               "      reflectance = transferFunction(data).rgb;\n"
               "   else if (reflectanceSource == 2)\n" // from scalar field
               "      reflectance = transferFunction(texture(texColormap, uv).r).rgb;\n"
               "   float cos_theta_o = max(0, dot(wo, n));\n"
               "   return reflectance / 3.1415926535 * cos_theta_o;\n"
               "}\n";
        // clang-format on
    }
}
