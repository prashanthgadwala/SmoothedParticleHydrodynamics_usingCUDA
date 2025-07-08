#include <vislab/opengl/scene_gl.hpp>

#include <vislab/opengl/actor_gl.hpp>
#include <vislab/opengl/bsdf_gl.hpp>
#include <vislab/opengl/colormap_texture_gl.hpp>
#include <vislab/opengl/diffuse_bsdf_gl.hpp>
#include <vislab/opengl/geometry_gl.hpp>
#include <vislab/opengl/light_gl.hpp>
#include <vislab/opengl/point_light_gl.hpp>
#include <vislab/opengl/projective_camera_gl.hpp>
#include <vislab/opengl/rectangle_geometry_gl.hpp>
#include <vislab/opengl/sphere_geometry_gl.hpp>
#include <vislab/opengl/transform_gl.hpp>
#include <vislab/opengl/trimesh_geometry_gl.hpp>

#include <vislab/graphics/actor.hpp>
#include <vislab/graphics/area_light.hpp>
#include <vislab/graphics/bsdf.hpp>
#include <vislab/graphics/colormap_texture.hpp>
#include <vislab/graphics/dielectric_bsdf.hpp>
#include <vislab/graphics/diffuse_bsdf.hpp>
#include <vislab/graphics/geometry.hpp>
#include <vislab/graphics/light.hpp>
#include <vislab/graphics/point_light.hpp>
#include <vislab/graphics/projective_camera.hpp>
#include <vislab/graphics/rectangle_geometry.hpp>
#include <vislab/graphics/scene.hpp>
#include <vislab/graphics/sphere_geometry.hpp>
#include <vislab/graphics/transform.hpp>
#include <vislab/graphics/trimesh_geometry.hpp>

namespace vislab
{
    std::shared_ptr<ActorGl> SceneGl::allocateActor(std::shared_ptr<const Resource> object, const RendererGl* renderer, OpenGL* opengl)
    {
        // TODO: Brute force way... This could later be done with reflection.

        // actor
        if (auto base = std::dynamic_pointer_cast<const Actor>(object))
        {
            auto result = std::make_shared<ActorGl>(base);
            if (!result->createDevice(opengl))
                return nullptr;
            return result;
        }

        return nullptr;
    }

    std::shared_ptr<ResourceBaseGl> SceneGl::allocateResource(std::shared_ptr<const Resource> object, OpenGL* opengl)
    {
        // TODO: Brute force way... This could later be done with reflection.

        // transform
        if (auto base = std::dynamic_pointer_cast<const Transform>(object))
        {
            auto result = std::make_shared<TransformGl>(base);
            if (!result->createDevice(opengl))
                return nullptr;
            return result;
        }

        // point light
        if (auto base = std::dynamic_pointer_cast<const PointLight>(object))
        {
            auto result = std::make_shared<PointLightGl>(base);
            if (!result->createDevice(opengl))
                return nullptr;
            return result;
        }

        // diffuse bsdf
        if (auto base = std::dynamic_pointer_cast<const DiffuseBSDF>(object))
        {
            auto result = std::make_shared<DiffuseBSDFGl>(base);
            if (!result->createDevice(opengl))
                return nullptr;
            return result;
        }

        // rectangle geometry
        if (auto base = std::dynamic_pointer_cast<const RectangleGeometry>(object))
        {
            auto result = std::make_shared<RectangleGeometryGl>(base);
            if (!result->createDevice(opengl))
                return nullptr;
            return result;
        }

        // sphere geometry
        if (auto base = std::dynamic_pointer_cast<const SphereGeometry>(object))
        {
            auto result = std::make_shared<SphereGeometryGl>(base);
            if (!result->createDevice(opengl))
                return nullptr;
            return result;
        }

        // trimesh geometry
        if (auto base = std::dynamic_pointer_cast<const TrimeshGeometry>(object))
        {
            auto result = std::make_shared<TrimeshGeometryGl>(base);
            if (!result->createDevice(opengl))
                return nullptr;
            return result;
        }

        // projective camera
        if (auto base = std::dynamic_pointer_cast<const ProjectiveCamera>(object))
        {
            auto result = std::make_shared<ProjectiveCameraGl>(base);
            if (!result->createDevice(opengl))
                return nullptr;
            return result;
        }

        // colormap texture
        if (auto base = std::dynamic_pointer_cast<const ColormapTexture>(object))
        {
            auto result = std::make_shared<ColormapTextureGl>(base);
            if (!result->createDevice(opengl))
                return nullptr;
            return result;
        }

        return nullptr;
    }

    void SceneGl::update(OpenGL* opengl, const Scene* scene, const RendererGl* renderer)
    {
        // delete out-dated resources
        for (auto it = begin(mResources); it != end(mResources);)
        {
            if (it->second->expired())
            {
                it = mResources.erase(it);
            }
            else
                ++it;
        }

        // clear the linear actor and light list, which we rebuild in the following.
        mLinearGeometries.clear();
        mLinearLights.clear();

        // get the scene
        bool anyTransformUpdated = false;
        if (scene)
        {
            // check all actors
            for (auto& actor : scene->actors)
            {
                // get the wrapped actor
                auto [actor_gl, new_resource] = getActor<ActorGl>(actor, renderer, opengl); // this creates the wrapper implicitly
                if (new_resource || actor->tags.has<ActorChangedTag>())
                {
                    actor_gl->update(this, renderer, opengl);
                }

                // if an actor was added, we rebuild the top-level acceleration data structure
                if (new_resource)
                {
                    anyTransformUpdated = true;
                }

                // if the actor has a geometry, add to the list of geometries, for which the acceleration data structure is built
                if (actor->components.has<Geometry>())
                {
                    mLinearGeometries.push_back(actor_gl.get());
                    if (actor->components.get<Geometry>()->hasChanged())
                        actor_gl->geometry->update(this, opengl);
                }

                // if the actor has a BSDF, see if it needs to get updated
                if (actor->components.has<BSDF>())
                {
                    if (actor->components.get<BSDF>()->hasChanged())
                        actor_gl->bsdf->update(this, opengl);
                }

                // if the actor has a light source, add it to the list of light sources
                if (actor->components.has<Light>())
                {
                    mLinearLights.push_back(actor_gl.get());
                    if (actor->components.get<Light>()->hasChanged())
                        actor_gl->light->update(this, opengl);
                }

                // has a transformation component been updated? if so, recompute the top-level acceleration data structure
                if (actor->components.has<Transform>())
                {
                    if (actor->components.get<Transform>()->hasChanged())
                    {
                        actor_gl->transform->update(this, opengl);
                        anyTransformUpdated = true;
                    }
                }

                // if the actor has a camera, see if it needs to get updated.
                if (actor->components.has<Camera>())
                {
                    if (actor->components.get<Camera>()->hasChanged())
                        actor_gl->camera->update(this, opengl);
                }
            }
        }

        // TODO: top-level acceleration data structure for frustum culling
    }

    const std::vector<ActorGl*>& SceneGl::linearGeometries() const
    {
        return mLinearGeometries;
    }

    const std::vector<ActorGl*>& SceneGl::linearLights() const
    {
        return mLinearLights;
    }
}
