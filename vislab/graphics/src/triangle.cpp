//#include <vislab/graphics/triangle.hpp>
//
//#include <vislab/graphics/mesh.hpp>
//
//#include <vislab/core/iarchive.hpp>
//
//namespace vislab
//{
//    Triangle::Triangle()
//        : mInverseSurfaceArea(0)
//    {
//        // update the internal precomputations whenever the matrix changed.
//        this->transform.onChanged += [this](Transform* sender, const Eigen::Matrix4d* args)
//        {
//            this->update();
//        };
//        update();
//    }
//
//    void Triangle::update()
//    {
//        mInverseSurfaceArea = 1. / surfaceArea();
//    }
//
//    Eigen::AlignedBox3d Triangle::objectBounds() const
//    {
//        return mesh->objectBounds();
//    }
//
//    double Triangle::surfaceArea() const
//    {
//        if (!mesh)
//            return 0;
//        return mesh->surfaceArea();
//    }
//
//    PositionSample Triangle::samplePosition(const Eigen::Vector2d& sample) const
//    {
//        return mesh->samplePosition(sample);
//    }
//
//    double Triangle::pdfPosition(const PositionSample& ps) const
//    {
//        return mesh->pdfPosition(ps);
//    }
//
//    DirectionSample Triangle::sampleDirection(const Interaction& it, const Eigen::Vector2d& sample) const
//    {
//        return mesh->sampleDirection(it, sample);
//    }
//
//    double Triangle::pdfDirection(const Interaction& it, const DirectionSample& ds) const
//    {
//        return mesh->pdfDirection(it, ds);
//    }
//
//    PreliminaryIntersection Triangle::preliminaryHit(const Ray3d& ray_) const
//    {
//        Ray3d ray                  = transform.transformRayInverse(ray_);
//        PreliminaryIntersection pi = mesh->preliminaryHit(ray);
//        pi.shape                   = this;
//        return pi;
//    }
//
//    SurfaceInteraction Triangle::computeSurfaceInteraction(const Ray3d& ray_, const PreliminaryIntersection& pi, EHitComputeFlag flags) const
//    {
//        Ray3d ray             = transform.transformRayInverse(ray_);
//        SurfaceInteraction si = mesh->computeSurfaceInteraction(ray, pi, flags);
//        si.position           = transform.transformPoint(si.position);
//        si.normal             = transform.transformNormal(si.normal);
//        si.sh_frame.n         = transform.transformNormal(si.sh_frame.n);
//        return si;
//    }
//
//    void Triangle::serialize(IArchive& archive)
//    {
//        archive("Mesh", mesh);
//    }
//
//    void Triangle::buildAccelerationTree()
//    {
//        mesh->buildAccelerationTree();
//    }
//}
