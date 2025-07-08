#include <vislab/opengl/geometry_gl.hpp>

#include <vislab/graphics/geometry.hpp>

namespace vislab
{
    GeometryGl::GeometryGl(std::shared_ptr<const Geometry> geometry, const ShaderSourceGl& _sourceCode, bool _writesDepth)
        : Interface<GeometryGl, ResourceGl<Geometry>>(geometry)
        , sourceCode(_sourceCode)
        , writesDepth(_writesDepth)
    {
    }
}
