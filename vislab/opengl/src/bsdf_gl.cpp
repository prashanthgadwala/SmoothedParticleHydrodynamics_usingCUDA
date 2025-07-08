#include <vislab/opengl/bsdf_gl.hpp>

namespace vislab
{
    BSDFGl::BSDFGl(std::shared_ptr<const BSDF> bsdf, const std::string& shaderCode)
        : Interface<BSDFGl, ResourceGl<BSDF>>(bsdf)
        , mShaderCode(shaderCode)
    {
    }

    const std::string& BSDFGl::getShaderCode() const
    {
        return mShaderCode;
    }
}
