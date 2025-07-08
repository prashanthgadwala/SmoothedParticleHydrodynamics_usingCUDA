#include <vislab/graphics/trimesh_geometry.hpp>

#include <vislab/core/iarchive.hpp>
#include <vislab/geometry/surfaces.hpp>
#include <vislab/geometry/vertex_normals.hpp>

namespace vislab
{
    TrimeshGeometry::TrimeshGeometry()
        : indices(nullptr)
        , positions(nullptr)
        , normals(nullptr)
        , texCoords(nullptr)
        , data(nullptr)
    {
        mBoundingBox.setEmpty();
    }

    TrimeshGeometry::TrimeshGeometry(std::shared_ptr<const Surfaces3f> surfaces)
        : indices(nullptr)
        , positions(nullptr)
        , normals(nullptr)
        , texCoords(nullptr)
        , data(nullptr)
    {
        copyFromSurfaces(surfaces);
        mBoundingBox = surfaces->getBoundingBox();
    }

    Eigen::AlignedBox3d TrimeshGeometry::objectBounds() const
    {
        return mBoundingBox;
    }

    void TrimeshGeometry::recomputeBoundingBox()
    {
        mBoundingBox.setEmpty();
        for (std::size_t i = 0; i < positions->getSize(); ++i)
            mBoundingBox.extend(positions->getValue(i).cast<double>());
    }

    void TrimeshGeometry::serialize(IArchive& archive)
    {
        archive("positions", positions);
        archive("normals", normals);
        archive("texCoords", texCoords);
        archive("indices", indices);
        archive("mBoundingBox", mBoundingBox);
    }

    void TrimeshGeometry::copyFromSurfaces(std::shared_ptr<const Surfaces3f> surfaces)
    {
        // no surfaces
        if (surfaces->getNumSurfaces() == 0)
            return;

        // allocate indices and positions (always required)
        if (indices)
            indices->clear();
        else
            indices = std::make_shared<Array3u>();
        if (positions)
            positions->clear();
        else
            positions = std::make_shared<Array3f>();

        // allocate optional arrays if necessary
        if (surfaces->getSurface(0)->normals)
        {
            if (normals)
                normals->clear();
            else
                normals = std::make_shared<Array3f>();
        }
        else
            normals = nullptr;
        if (surfaces->getSurface(0)->texCoords)
        {
            if (texCoords)
                texCoords->clear();
            else
                texCoords = std::make_shared<Array2f>();
        }
        else
            texCoords = nullptr;

        // convert surfaces to linear consecutive data structure (triangle list)
        Eigen::Index indexCounter = 0, positionCounter = 0, normalCounter = 0, texcoordCounter = 0;
        bool hasNormals = false;
        mBoundingBox.setEmpty();
        for (int isurf = 0; isurf < surfaces->getNumSurfaces(); ++isurf)
        {
            auto surface                    = surfaces->getSurface(isurf);
            Eigen::Index numSurfaceIndices  = surface->indices->getSize();
            Eigen::Index numSurfaceVertices = surface->positions->getSize();
            switch (surface->primitiveTopology)
            {
            case EPrimitiveTopology::TriangleList:
            {
                // copy index buffer
                Eigen::Index numPrimitives = numSurfaceIndices / 3;
                indices->setSize(indices->getSize() + numPrimitives);
                Eigen::Index baseoffset = positionCounter;
                for (Eigen::Index index = 0; index < numPrimitives; ++index)
                {
                    indices->setValue(indexCounter++, Eigen::Vector3u(
                                                          baseoffset + surface->indices->getValue(index * 3 + 0).x(),
                                                          baseoffset + surface->indices->getValue(index * 3 + 1).x(),
                                                          baseoffset + surface->indices->getValue(index * 3 + 2).x()));
                }

                // copy positions
                positions->setSize(positions->getSize() + numSurfaceVertices);
                for (Eigen::Index vertex = 0; vertex < numSurfaceVertices; ++vertex)
                {
                    auto& pos = surface->positions->getValue(vertex);
                    positions->setValue(positionCounter++, pos);
                    mBoundingBox.extend(pos.cast<double>());
                }

                // copy normals
                auto normalArray = surface->normals;
                if (normalArray)
                {
                    hasNormals = true;
                    normals->setSize(normals->getSize() + numSurfaceVertices);
                    for (Eigen::Index vertex = 0; vertex < numSurfaceVertices; ++vertex)
                    {
                        normals->setValue(
                            normalCounter++,
                            normalArray->getValue(vertex));
                    }
                }

                // copy texture coordinates
                auto texcoordArray = texCoords;
                if (texcoordArray)
                {
                    texCoords->setSize(texCoords->getSize() + numSurfaceVertices);
                    for (Eigen::Index vertex = 0; vertex < numSurfaceVertices; ++vertex)
                    {
                        texCoords->setValue(
                            texcoordCounter++,
                            texcoordArray->getValue(vertex));
                    }
                }
                break;
            }
            default:
                throw std::logic_error("Primitive type not implemented.");
            }
        }

        // if there were no normals provided, then compute vertex normals
        if (!hasNormals)
        {
            normals = std::make_shared<Array3f>();
            VertexNormals3f::computeNormals(positions, indices, normals);
        }
    }
}
