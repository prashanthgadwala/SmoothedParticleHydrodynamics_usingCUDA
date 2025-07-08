#include <vislab/geometry/vertex_normals.hpp>

namespace vislab
{

    UpdateInfo VertexNormals3f::internalUpdate(ProgressInfo& progress)
    {
        // get input surface and output surface
        auto inSurfaces  = inputSurfaces.getData();
        auto outSurfaces = outputSurfaces.getData();

        // for each input surface
        for (std::size_t iSurface = 0; iSurface < inSurfaces->getNumSurfaces(); ++iSurface)
        {
            // get the input surface
            auto inSurface = inSurfaces->getSurface(iSurface);

            // switch depending on primitive topology
            if (inSurface->primitiveTopology != EPrimitiveTopology::TriangleList)
            {
                return UpdateInfo::reportError("Algorithm not implemented for primitive topology.");
            }

            // is there an index buffer?
            if (inSurface->positions->getSize() > 0 && inSurface->indices->getSize() == 0)
            {
                return UpdateInfo::reportWarning("Surface " + std::to_string(iSurface) + " with positions, but without indices.");
            }

            // create output surface
            auto outSurface = outSurfaces->createSurface();

            // copy positions, indices, attributes, and optional texcoords
            outSurface->primitiveTopology = EPrimitiveTopology::TriangleList;
            outSurface->positions         = inSurface->positions->clone();
            outSurface->indices           = inSurface->indices->clone();
            outSurface->attributes        = inSurface->attributes->clone();
            if (inSurface->texCoords)
                outSurface->texCoords = inSurface->texCoords->clone();
        }

        // invoke the vertex normal computation
        computeNormals(outSurfaces);

        // compute the new bounding box
        outSurfaces->recomputeBoundingBox();

        progress.allJobsDone();
        return UpdateInfo::reportValid();
    }

    void VertexNormals3f::computeNormals(std::shared_ptr<const Array3f> positions, std::shared_ptr<const Array1u> indices, std::shared_ptr<Array3f> normals)
    {
        // resize the normal buffer (nothing happens if it already has the right size)
        normals->setSize(positions->getSize());

        // clear the normals to zero.
        normals->setZero();

        // for each triangle
        for (Eigen::Index i = 0; i < indices->getSize() / 3; ++i)
        {
            // get vertex positions
            uint32_t i0        = indices->getValue(3 * i + 0).x();
            uint32_t i1        = indices->getValue(3 * i + 1).x();
            uint32_t i2        = indices->getValue(3 * i + 2).x();
            Eigen::Vector3f p0 = positions->getValue(i0);
            Eigen::Vector3f p1 = positions->getValue(i1);
            Eigen::Vector3f p2 = positions->getValue(i2);

            // compute face normal (area weighted)
            Eigen::Vector3f n = (p1 - p0).cross(p2 - p0);

            // sum up on vertices
            normals->setValue(i0, normals->getValue(i0) + n);
            normals->setValue(i1, normals->getValue(i1) + n);
            normals->setValue(i2, normals->getValue(i2) + n);
        }

        // normalize all normals (area-weighted averaging)
        for (Eigen::Index i = 0; i < normals->getSize(); ++i)
        {
            Eigen::Vector3f n = normals->getValue(i);
            normals->setValue(i, n.stableNormalized());
        }
    }

    void VertexNormals3f::computeNormals(std::shared_ptr<const Array3f> positions, std::shared_ptr<const Array3u> indices, std::shared_ptr<Array3f> normals)
    {
        // resize the normal buffer (nothing happens if it already has the right size)
        normals->setSize(positions->getSize());

        // clear the normals to zero.
        normals->setZero();

        // for each triangle
        for (Eigen::Index i = 0; i < indices->getSize(); ++i)
        {
            // get vertex positions
            Eigen::Vector3u index = indices->getValue(i);
            Eigen::Vector3f p0    = positions->getValue(index.x());
            Eigen::Vector3f p1    = positions->getValue(index.y());
            Eigen::Vector3f p2    = positions->getValue(index.z());

            // compute face normal (area weighted)
            Eigen::Vector3f n = (p1 - p0).cross(p2 - p0);

            // sum up on vertices
            normals->setValue(index.x(), normals->getValue(index.x()) + n);
            normals->setValue(index.y(), normals->getValue(index.y()) + n);
            normals->setValue(index.z(), normals->getValue(index.z()) + n);
        }

        // normalize all normals (area-weighted averaging)
        for (Eigen::Index i = 0; i < normals->getSize(); ++i)
        {
            Eigen::Vector3f n = normals->getValue(i);
            normals->setValue(i, n.stableNormalized());
        }
    }

    void VertexNormals3f::computeNormals(std::shared_ptr<Surface3f> surface)
    {
        // if there is no normal buffer yet, allocate one.
        if (!surface->normals)
        {
            surface->normals = std::make_shared<Array3f>();
        }
        computeNormals(surface->positions, surface->indices, surface->normals);
    }

    void VertexNormals3f::computeNormals(std::shared_ptr<Surfaces3f> surfaces)
    {
        std::size_t numSurfaces = surfaces->getNumSurfaces();
        for (std::size_t isurface = 0; isurface < numSurfaces; ++isurface)
        {
            computeNormals(surfaces->getSurface(isurface));
        }
    }

}
