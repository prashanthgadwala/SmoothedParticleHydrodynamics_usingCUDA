#include "init_opengl.hpp"

#include <vislab/core/reflect.hpp>

#include <vislab/opengl/bsdf_gl.hpp>
#include <vislab/opengl/camera_gl.hpp>
#include <vislab/opengl/diffuse_bsdf_gl.hpp>
#include <vislab/opengl/forward_renderer_gl.hpp>
#include <vislab/opengl/geometry_gl.hpp>
#include <vislab/opengl/light_gl.hpp>
#include <vislab/opengl/point_light_gl.hpp>
#include <vislab/opengl/projective_camera_gl.hpp>
#include <vislab/opengl/rectangle_geometry_gl.hpp>
#include <vislab/opengl/resource_base_gl.hpp>
#include <vislab/opengl/resource_gl.hpp>
#include <vislab/opengl/transform_gl.hpp>
#include <vislab/opengl/trimesh_geometry_gl.hpp>

void init_opengl()
{
    using namespace vislab;

    // Interfaces

    reflect<ResourceBaseGl, Object>("ResourceBaseGl");
    reflect<ResourceGl<Camera>, ResourceBaseGl>("ResourceGl<Camera>");
    reflect<ResourceGl<Light>, ResourceBaseGl>("ResourceGl<Light>");
    reflect<ResourceGl<BSDF>, ResourceBaseGl>("ResourceGl<BSDF>");
    reflect<ResourceGl<Geometry>, ResourceBaseGl>("ResourceGl<Geometry>");
    reflect<ResourceGl<Transform>, ResourceBaseGl>("ResourceGl<Transform>");
    reflect<CameraGl, ResourceGl<Camera>>("CameraGl");
    reflect<LightGl, ResourceGl<Light>>("LightGl");
    reflect<BSDFGl, ResourceGl<BSDF>>("BSDFGl");
    reflect<GeometryGl, ResourceGl<Geometry>>("GeometryGl");

    // Constructibles

    reflect<vislab::ProjectiveCameraGl, CameraGl>("ProjectiveCameraGl");
    reflect<vislab::DiffuseBSDFGl, BSDFGl>("DiffuseBSDFGl");
    reflect<vislab::PointLightGl, LightGl>("PointLightGl");
    reflect<vislab::RectangleGeometryGl, GeometryGl>("RectangleGeometryGl");
    reflect<vislab::TrimeshGeometryGl, GeometryGl>("TrimeshGeometryGl");
    reflect<vislab::TransformGl, ResourceGl<Transform>>("TransformGl");
    reflect<vislab::ForwardRendererGl, Renderer>("ForwardRendererGl");
}
