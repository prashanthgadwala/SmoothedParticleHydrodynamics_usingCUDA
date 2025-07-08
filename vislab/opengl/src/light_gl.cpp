#include <vislab/opengl/light_gl.hpp>

namespace vislab
{
    LightGl::LightGl(std::shared_ptr<const Light> light)
        : Interface<LightGl, ResourceGl<Light>>(light)
    {
    }
}
