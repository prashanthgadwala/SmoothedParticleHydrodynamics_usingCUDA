#include <vislab/graphics/scene.hpp>

// #include <vislab/graphics/camera.hpp>
// #include <vislab/graphics/light.hpp>
// #include <vislab/graphics/math.hpp>
// #include <vislab/graphics/preliminary_interaction.hpp>
#include <vislab/graphics/actor.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    // SurfaceInteraction Scene::intersect(const Ray3d& ray) const
    //{
    //     // test all objects (linear for now...)
    //     PreliminaryIntersection pi;
    //     if (mBoundingVolumeHierarchy)
    //     {
    //         pi = mBoundingVolumeHierarchy->preliminaryHit(ray);
    //     }
    //     else
    //     {
    //         for (std::size_t i = 0; i < shapes.size(); ++i)
    //         {
    //             auto preliminaryHit = shapes[i]->preliminaryHit(ray);
    //             if (preliminaryHit.t < pi.t)
    //             {
    //                 pi = preliminaryHit;
    //             }
    //         }
    //     }

    //    // if there was no hit, return invalid surface interaction.
    //    if (!pi.success())
    //    {
    //        SurfaceInteraction si;
    //        si.wi = -ray.direction;
    //        si.t  = std::numeric_limits<double>::infinity();
    //        return si;
    //    }

    //    // compute detailed surface interaction for the closest hit (if there is one)
    //    return pi.computeSurfaceInteraction(ray, EHitComputeFlag::All);
    //}

    // bool Scene::anyHit(const Ray3d& ray) const
    //{
    //     if (mBoundingVolumeHierarchy)
    //     {
    //         return mBoundingVolumeHierarchy->anyHit(ray);
    //     }
    //     else
    //     {
    //         // test all objects
    //         for (std::size_t i = 0; i < shapes.size(); ++i)
    //         {
    //             if (shapes[i]->anyHit(ray))
    //                 return true;
    //         }
    //         return false;
    //     }
    // }

    // std::pair<DirectionSample, Spectrum> Scene::sampleLightDirection(const Interaction& it, const Eigen::Vector2d& sample_, bool test_visibility) const
    //{
    //     Eigen::Vector2d sample(sample_);
    //     DirectionSample ds;
    //     Spectrum spec = Spectrum::Zero();

    //    if (!lights.empty())
    //    {
    //        if (lights.size() == 1)
    //        {
    //            // Fast path if there is only one light
    //            std::tie(ds, spec) = lights[0]->sampleDirection(it, sample);
    //        }
    //        else
    //        {
    //            double light_pdf = 1. / lights.size();

    //            // Randomly pick an light
    //            uint32_t index =
    //                std::min(uint32_t(sample.x() * (double)lights.size()),
    //                         (uint32_t)lights.size() - 1);

    //            // Rescale sample.x() to lie in [0,1) again
    //            sample.x() = (sample.x() - index * light_pdf) * lights.size();

    //            auto light = lights[index];

    //            // Sample a direction towards the light
    //            std::tie(ds, spec) = light->sampleDirection(it, sample);

    //            // Account for the discrete probability of sampling this light
    //            ds.pdf *= light_pdf;
    //            spec *= 1.0 / light_pdf;
    //        }

    //        // Perform a visibility test if requested
    //        if (test_visibility && ds.pdf != 0.f)
    //        {
    //            Ray ray(it.position, ds.direction, rayEpsilon() * (1. + it.position.cwiseAbs().maxCoeff()),
    //                    ds.dist * (1. - shadowEpsilon()));

    //            if (anyHit(ray))
    //                spec.setZero();
    //        }
    //    }
    //    else
    //    {
    //        ds = DirectionSample();
    //        spec.setZero();
    //    }

    //    return { ds, spec };
    //}

    // double Scene::pdfLightDirection(const Interaction& ref, const DirectionSample& ds) const
    //{
    //     double light_pdf = 1. / lights.size();
    //     return ds.light->pdfDirection(ref, ds) * light_pdf;
    // }

    void Scene::serialize(IArchive& archive)
    {
        archive("Actors", actors);
        // archive("Lights", lights);
        // archive("Environment", environment);
        // archive("Camera", camera);
    }

    std::shared_ptr<const Actor> Scene::getActor(const std::string& name) const
    {
        for (std::size_t i = 0; i < actors.size(); ++i)
            if (actors[i]->name == name)
                return actors[i];
        return nullptr;
    }

    std::shared_ptr<Actor> Scene::getActor(const std::string& name)
    {
        for (std::size_t i = 0; i < actors.size(); ++i)
            if (actors[i]->name == name)
                return actors[i];
        return nullptr;
    }

    /*void Scene::buildAccelerationTree()
    {
        mBoundingVolumeHierarchy = std::make_shared<BoundingVolumeHierarchy<Shape, 3>>();
        mBoundingVolumeHierarchy->getShapes().resize(shapes.size());
        for (std::size_t i = 0; i < shapes.size(); ++i)
        {
            shapes[i]->buildAccelerationTree();
            mBoundingVolumeHierarchy->getShapes()[i] = shapes[i].get();
        }
        mBoundingVolumeHierarchy->build();
    }*/
}
