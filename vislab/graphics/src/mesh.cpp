//#include <vislab/graphics/mesh.hpp>
//
//#include <vislab/graphics/math.hpp>
//#include <vislab/graphics/warp.hpp>
//
//#include <vislab/core/array.hpp>
//
//namespace vislab
//{
//    Mesh::Mesh()
//    {
//    }
//
//    Eigen::AlignedBox3d Mesh::objectBounds() const
//    {
//        Eigen::AlignedBox3d bounds;
//        bounds.setEmpty();
//        if (positions)
//        {
//            for (Eigen::Index i = 0; i < positions->getSize(); ++i)
//                bounds.extend(positions->getValue(i).cast<double>());
//        }
//        return bounds;
//    }
//
//    double Mesh::surfaceArea() const
//    {
//        double area = 0;
//        if (!indices || !positions)
//            return area;
//        for (Eigen::Index i = 0; i < indices->getSize(); ++i)
//        {
//            Eigen::Vector3u fi = indices->getValue(i);
//            Eigen::Vector3d p0 = positions->getValue(fi[0]).cast<double>();
//            Eigen::Vector3d p1 = positions->getValue(fi[1]).cast<double>();
//            Eigen::Vector3d p2 = positions->getValue(fi[2]).cast<double>();
//
//            area += 0.5 * (p1 - p0).cross(p2 - p0).norm();
//        }
//        return area;
//    }
//
//    PositionSample Mesh::samplePosition(const Eigen::Vector2d& sample) const
//    {
//        throw std::logic_error("not implemented");
//        PositionSample ps;
//        return ps;
//    }
//
//    double Mesh::pdfPosition(const PositionSample& ps) const
//    {
//        throw std::logic_error("not implemented");
//        return 0;
//    }
//
//    DirectionSample Mesh::sampleDirection(const Interaction& it, const Eigen::Vector2d& sample) const
//    {
//        throw std::logic_error("not implemented");
//        DirectionSample result;
//        return result;
//    }
//
//    double Mesh::pdfDirection(const Interaction& it, const DirectionSample& ds) const
//    {
//        throw std::logic_error("not implemented");
//        return 0;
//    }
//
//    PreliminaryIntersection Mesh::preliminaryHit(const Ray3d& _ray) const
//    {
//        if (mBoundingVolumeHierarchy)
//        {
//            return mBoundingVolumeHierarchy->preliminaryHit(_ray);
//        }
//        else
//        {
//            // test all objects (linear for now...)
//            Ray3d ray = _ray;
//            PreliminaryIntersection pi;
//            for (Eigen::Index i = 0; i < indices->getSize(); ++i)
//            {
//                auto preliminaryHit = rayTriangleIntersection(i, ray);
//                if (preliminaryHit.isValid() &&
//                    ray.tMin < preliminaryHit.t &&
//                    preliminaryHit.t <= ray.tMax &&
//                    preliminaryHit.t < pi.t)
//                {
//                    pi       = preliminaryHit;
//                    ray.tMax = pi.t;
//                }
//            }
//            return pi;
//        }
//    }
//
//    SurfaceInteraction Mesh::computeSurfaceInteraction(const Ray3d& ray, const PreliminaryIntersection& pi, EHitComputeFlag flags) const
//    {
//        bool active = pi.isValid();
//
//        double b1 = pi.prim_uv.x();
//        double b2 = pi.prim_uv.y();
//        double b0 = 1.f - b1 - b2;
//
//        Eigen::Vector3u fi = indices->getValue(pi.prim_index);
//        Eigen::Vector3d p0 = positions->getValue(fi[0]).cast<double>();
//        Eigen::Vector3d p1 = positions->getValue(fi[1]).cast<double>();
//        Eigen::Vector3d p2 = positions->getValue(fi[2]).cast<double>();
//
//        Eigen::Vector3d dp0 = p1 - p0,
//                        dp1 = p2 - p0;
//
//        SurfaceInteraction si;
//        si.t = active ? pi.t : std::numeric_limits<double>::infinity();
//
//        // Re-interpolate intersection using barycentric coordinates
//        si.position = (p0 * b0 + p1 * b1 + p2 * b2);
//
//        // Face normal
//        si.normal = dp0.cross(dp1).normalized();
//
//        // Texture coordinates (if available)
//        si.uv                        = Eigen::Vector2d(b1, b2);
//        std::tie(si.dp_du, si.dp_dv) = coordinateSystem(si.normal);
//        if (texCoords && hasFlag(flags, EHitComputeFlag::UV))
//        {
//            Eigen::Vector2d uv0 = texCoords->getValue(fi[0]).cast<double>();
//            Eigen::Vector2d uv1 = texCoords->getValue(fi[1]).cast<double>();
//            Eigen::Vector2d uv2 = texCoords->getValue(fi[2]).cast<double>();
//
//            si.uv = uv0 * b0 + uv1 * b1 + uv2 * b2;
//
//            if (hasFlag(flags, EHitComputeFlag::dPdUV))
//            {
//                Eigen::Vector2d duv0 = uv1 - uv0,
//                                duv1 = uv2 - uv0;
//
//                double det     = duv0.x() * duv1.y() - duv0.y() * duv1.x(),
//                       inv_det = 1. / det;
//
//                bool valid = det != 0.;
//
//                si.dp_du = (duv1.y() * dp0 - duv0.y() * dp1) * inv_det;
//                si.dp_dv = (-duv1.x() * dp0 + duv0.x() * dp1) * inv_det;
//            }
//        }
//
//        // Shading normal (if available)
//        if (normals && hasFlag(flags, EHitComputeFlag::ShadingFrame))
//        {
//            Eigen::Vector3d n0 = normals->getValue(fi[0]).cast<double>();
//            Eigen::Vector3d n1 = normals->getValue(fi[1]).cast<double>();
//            Eigen::Vector3d n2 = normals->getValue(fi[2]).cast<double>();
//
//            si.sh_frame.n = (n0 * b0 + n1 * b1 + n2 * b2).normalized();
//
//            // si.dn_du = si.dn_dv = zero<Vector3f>();
//            // if (has_flag(flags, EHitComputeFlag::dNSdUV))
//            //{
//            //     /* Now compute the derivative of "normalize(u*n1 + v*n2 + (1-u-v)*n0)"
//            //        with respect to [u, v] in the local triangle parameterization.
//
//            //       Since d/du [f(u)/|f(u)|] = [d/du f(u)]/|f(u)|
//            //           - f(u)/|f(u)|^3 <f(u), d/du f(u)>, this results in
//            //    */
//
//            //    Normal3f N = b0 * n1 + b1 * n2 + b2 * n0;
//            //    Float il   = rsqrt(squared_norm(N));
//            //    N *= il;
//
//            //    si.dn_du = (n1 - n0) * il;
//            //    si.dn_dv = (n2 - n0) * il;
//
//            //    si.dn_du = fnmadd(N, dot(N, si.dn_du), si.dn_du);
//            //    si.dn_dv = fnmadd(N, dot(N, si.dn_dv), si.dn_dv);
//            //}
//        }
//        else
//        {
//            si.sh_frame.n = si.normal;
//        }
//
//        return si;
//    }
//
//    PreliminaryIntersection Mesh::rayTriangleIntersection(const uint32_t& faceIndex, const Ray3d& ray) const
//    {
//        Eigen::Vector3u fi = indices->getValue(faceIndex);
//        Eigen::Vector3d p0 = positions->getValue(fi[0]).cast<double>();
//        Eigen::Vector3d p1 = positions->getValue(fi[1]).cast<double>();
//        Eigen::Vector3d p2 = positions->getValue(fi[2]).cast<double>();
//
//        Eigen::Vector3d e1 = p1 - p0, e2 = p2 - p0;
//
//        Eigen::Vector3d pvec = ray.direction.cross(e2);
//        double inv_det       = 1. / (e1.dot(pvec));
//
//        Eigen::Vector3d tvec = ray.origin - p0;
//        double u             = tvec.dot(pvec) * inv_det;
//        bool active          = u >= 0. && u <= 1.;
//
//        Eigen::Vector3d qvec = tvec.cross(e1);
//        double v             = ray.direction.dot(qvec) * inv_det;
//        active &= v >= 0. && u + v <= 1.;
//
//        double t = e2.dot(qvec) * inv_det;
//        active &= t >= ray.tMin && t <= ray.tMax;
//
//        PreliminaryIntersection pi;
//        pi.t          = active ? t : std::numeric_limits<double>::infinity();
//        pi.prim_uv    = Eigen::Vector2d(u, v);
//        pi.prim_index = faceIndex;
//
//        return pi;
//    }
//
//    void Mesh::serialize(IArchive& archive)
//    {
//        archive("Positions", positions);
//        archive("Normals", normals);
//        archive("TexCoords", texCoords);
//        archive("Indices", indices);
//    }
//
//    void Mesh::recomputeVertexNormals()
//    {
//        // if there are no vertex positions, cancel immediately.
//        if (!positions)
//            return;
//
//        // if there is no normal buffer yet, allocate one.
//        if (!normals)
//        {
//            normals = std::make_shared<Array3f>();
//        }
//
//        // resize the normal buffer (nothing happens if it already has the right size)
//        normals->setSize(positions->getSize());
//
//        // clear the normals to zero.
//        normals->setZero();
//
//        // for each triangle
//        for (Eigen::Index i = 0; i < indices->getSize(); ++i)
//        {
//            // get vertex positions
//            Eigen::Vector3u fi = indices->getValue(i);
//            Eigen::Vector3d p0 = positions->getValue(fi[0]).cast<double>();
//            Eigen::Vector3d p1 = positions->getValue(fi[1]).cast<double>();
//            Eigen::Vector3d p2 = positions->getValue(fi[2]).cast<double>();
//
//            // compute face normal (area weighted)
//            Eigen::Vector3f n = (p1 - p0).cross(p2 - p0).cast<float>();
//
//            // sum up on vertices
//            normals->setValue(fi[0], normals->getValue(fi[0]) + n);
//            normals->setValue(fi[1], normals->getValue(fi[1]) + n);
//            normals->setValue(fi[2], normals->getValue(fi[2]) + n);
//        }
//
//        // normalize all normals (area-weighted averaging)
//        for (Eigen::Index i = 0; i < normals->getSize(); ++i)
//        {
//            Eigen::Vector3f n = normals->getValue(i);
//            normals->setValue(i, n.normalized());
//        }
//    }
//
//    void Mesh::buildAccelerationTree()
//    {
//        mTriangles.resize(indices->getSize());
//        for (std::size_t i = 0; i < mTriangles.size(); ++i)
//        {
//            mTriangles[i].face = i;
//            mTriangles[i].mesh = this;
//
//            Eigen::Vector3u fi = indices->getValue(i);
//            Eigen::Vector3d p0 = positions->getValue(fi[0]).cast<double>();
//            Eigen::Vector3d p1 = positions->getValue(fi[1]).cast<double>();
//            Eigen::Vector3d p2 = positions->getValue(fi[2]).cast<double>();
//            mTriangles[i].worldBoundingBox.setEmpty();
//            mTriangles[i].worldBoundingBox.extend(p0);
//            mTriangles[i].worldBoundingBox.extend(p1);
//            mTriangles[i].worldBoundingBox.extend(p2);
//        }
//        mBoundingVolumeHierarchy = std::make_shared<BoundingVolumeHierarchy<Triangle, 3>>();
//        mBoundingVolumeHierarchy->getShapes().resize(indices->getSize());
//        for (std::size_t i = 0; i < indices->getSize(); ++i)
//            mBoundingVolumeHierarchy->getShapes()[i] = &mTriangles[i];
//        mBoundingVolumeHierarchy->build();
//    }
//
//    Mesh::Triangle::Triangle()
//        : face(-1)
//        , mesh(NULL)
//    {
//
//    }
//
//    Eigen::AlignedBox3d Mesh::Triangle::worldBounds() const
//    {
//        return worldBoundingBox;
//    }
//
//    PreliminaryIntersection Mesh::Triangle::preliminaryHit(const Ray3d& ray) const
//    {
//        return mesh->rayTriangleIntersection(face, ray);
//    }
//}
