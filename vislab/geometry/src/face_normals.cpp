#include <vislab/geometry/face_normals.hpp>

namespace vislab
{
    /**
     * @brief Helper class that devirtualizes an iarray, which allows getting and setting values.
     */
    class DevirtualizedArray
    {
    public:
        /**
         * @brief Constructor that tries casting the incoming IArray into all kinds of arrays.
         * @param _iarray IArray to devirtualize.
         */
        DevirtualizedArray(std::shared_ptr<IArray> _iarray)
            : iarray(_iarray)
        {
            array1f       = std::dynamic_pointer_cast<Array1f>(iarray);
            array1d       = std::dynamic_pointer_cast<Array1d>(iarray);
            array1i_16    = std::dynamic_pointer_cast<Array1i_16>(iarray);
            array1i       = std::dynamic_pointer_cast<Array1i>(iarray);
            array1i_64    = std::dynamic_pointer_cast<Array1i_64>(iarray);
            array1u_16   = std::dynamic_pointer_cast<Array1u_16>(iarray);
            array1u      = std::dynamic_pointer_cast<Array1u>(iarray);
            array1u_64   = std::dynamic_pointer_cast<Array1u_64>(iarray);
            array2f       = std::dynamic_pointer_cast<Array2f>(iarray);
            array2d       = std::dynamic_pointer_cast<Array2d>(iarray);
            array2i_16    = std::dynamic_pointer_cast<Array2i_16>(iarray);
            array2i       = std::dynamic_pointer_cast<Array2i>(iarray);
            array2i_64    = std::dynamic_pointer_cast<Array2i_64>(iarray);
            array2u_16   = std::dynamic_pointer_cast<Array2u_16>(iarray);
            array2u      = std::dynamic_pointer_cast<Array2u>(iarray);
            array2u_64   = std::dynamic_pointer_cast<Array2u_64>(iarray);
            array3f       = std::dynamic_pointer_cast<Array3f>(iarray);
            array3d       = std::dynamic_pointer_cast<Array3d>(iarray);
            array3i_16    = std::dynamic_pointer_cast<Array3i_16>(iarray);
            array3i       = std::dynamic_pointer_cast<Array3i>(iarray);
            array3i_64    = std::dynamic_pointer_cast<Array3i_64>(iarray);
            array3u_16   = std::dynamic_pointer_cast<Array3u_16>(iarray);
            array3u      = std::dynamic_pointer_cast<Array3u>(iarray);
            array3u_64   = std::dynamic_pointer_cast<Array3u_64>(iarray);
            array4f       = std::dynamic_pointer_cast<Array4f>(iarray);
            array4d       = std::dynamic_pointer_cast<Array4d>(iarray);
            array4i_16    = std::dynamic_pointer_cast<Array4i_16>(iarray);
            array4i       = std::dynamic_pointer_cast<Array4i>(iarray);
            array4i_64    = std::dynamic_pointer_cast<Array4i_64>(iarray);
            array4u_16   = std::dynamic_pointer_cast<Array4u_16>(iarray);
            array4u      = std::dynamic_pointer_cast<Array4u>(iarray);
            array4u_64   = std::dynamic_pointer_cast<Array4u_64>(iarray);
            array2x2f     = std::dynamic_pointer_cast<Array2x2f>(iarray);
            array2x2d     = std::dynamic_pointer_cast<Array2x2d>(iarray);
            array2x2i_16  = std::dynamic_pointer_cast<Array2x2i_16>(iarray);
            array2x2i     = std::dynamic_pointer_cast<Array2x2i>(iarray);
            array2x2i_64  = std::dynamic_pointer_cast<Array2x2i_64>(iarray);
            array2x2u_16 = std::dynamic_pointer_cast<Array2x2u_16>(iarray);
            array2x2u    = std::dynamic_pointer_cast<Array2x2u>(iarray);
            array2x2u_64 = std::dynamic_pointer_cast<Array2x2u_64>(iarray);
            array3x3f     = std::dynamic_pointer_cast<Array3x3f>(iarray);
            array3x3d     = std::dynamic_pointer_cast<Array3x3d>(iarray);
            array3x3i_16  = std::dynamic_pointer_cast<Array3x3i_16>(iarray);
            array3x3i     = std::dynamic_pointer_cast<Array3x3i>(iarray);
            array3x3i_64  = std::dynamic_pointer_cast<Array3x3i_64>(iarray);
            array3x3u_16 = std::dynamic_pointer_cast<Array3x3u_16>(iarray);
            array3x3u    = std::dynamic_pointer_cast<Array3x3u>(iarray);
            array3x3u_64 = std::dynamic_pointer_cast<Array3x3u_64>(iarray);
            array4x4f     = std::dynamic_pointer_cast<Array4x4f>(iarray);
            array4x4d     = std::dynamic_pointer_cast<Array4x4d>(iarray);
            array4x4i_16  = std::dynamic_pointer_cast<Array4x4i_16>(iarray);
            array4x4i     = std::dynamic_pointer_cast<Array4x4i>(iarray);
            array4x4i_64  = std::dynamic_pointer_cast<Array4x4i_64>(iarray);
            array4x4u_16 = std::dynamic_pointer_cast<Array4x4u_16>(iarray);
            array4x4u    = std::dynamic_pointer_cast<Array4x4u>(iarray);
            array4x4u_64 = std::dynamic_pointer_cast<Array4x4u_64>(iarray);
        }

        /**
         * @brief Allocates an array of the type that this class is devirtualizing.
         * @return Allocated array of size 0.
         */
        std::shared_ptr<IArray> allocate()
        {
            if (array1f != nullptr)
                return std::make_shared<Array1f>();
            if (array1d != nullptr)
                return std::make_shared<Array1d>();
            if (array1i_16 != nullptr)
                return std::make_shared<Array1i_16>();
            if (array1i != nullptr)
                return std::make_shared<Array1i>();
            if (array1i_64 != nullptr)
                return std::make_shared<Array1i_64>();
            if (array1u_16 != nullptr)
                return std::make_shared<Array1u_16>();
            if (array1u != nullptr)
                return std::make_shared<Array1u>();
            if (array1u_64 != nullptr)
                return std::make_shared<Array1u_64>();
            if (array2f != nullptr)
                return std::make_shared<Array2f>();
            if (array2d != nullptr)
                return std::make_shared<Array2d>();
            if (array2i_16 != nullptr)
                return std::make_shared<Array2i_16>();
            if (array2i != nullptr)
                return std::make_shared<Array2i>();
            if (array2i_64 != nullptr)
                return std::make_shared<Array2i_64>();
            if (array2u_16 != nullptr)
                return std::make_shared<Array2u_16>();
            if (array2u != nullptr)
                return std::make_shared<Array2u>();
            if (array2u_64 != nullptr)
                return std::make_shared<Array2u_64>();
            if (array3f != nullptr)
                return std::make_shared<Array3f>();
            if (array3d != nullptr)
                return std::make_shared<Array3d>();
            if (array3i_16 != nullptr)
                return std::make_shared<Array3i_16>();
            if (array3i != nullptr)
                return std::make_shared<Array3i>();
            if (array3i_64 != nullptr)
                return std::make_shared<Array3i_64>();
            if (array3u_16 != nullptr)
                return std::make_shared<Array3u_16>();
            if (array3u != nullptr)
                return std::make_shared<Array3u>();
            if (array3u_64 != nullptr)
                return std::make_shared<Array3u_64>();
            if (array4f != nullptr)
                return std::make_shared<Array4f>();
            if (array4d != nullptr)
                return std::make_shared<Array4d>();
            if (array4i_16 != nullptr)
                return std::make_shared<Array4i_16>();
            if (array4i != nullptr)
                return std::make_shared<Array4i>();
            if (array4i_64 != nullptr)
                return std::make_shared<Array4i_64>();
            if (array4u_16 != nullptr)
                return std::make_shared<Array4u_16>();
            if (array4u != nullptr)
                return std::make_shared<Array4u>();
            if (array4u_64 != nullptr)
                return std::make_shared<Array4u_64>();
            if (array2x2f != nullptr)
                return std::make_shared<Array2x2f>();
            if (array2x2d != nullptr)
                return std::make_shared<Array2x2d>();
            if (array2x2i_16 != nullptr)
                return std::make_shared<Array2x2i_16>();
            if (array2x2i != nullptr)
                return std::make_shared<Array2x2i>();
            if (array2x2i_64 != nullptr)
                return std::make_shared<Array2x2i_64>();
            if (array2x2u_16 != nullptr)
                return std::make_shared<Array2x2u_16>();
            if (array2x2u != nullptr)
                return std::make_shared<Array2x2u>();
            if (array2x2u_64 != nullptr)
                return std::make_shared<Array2x2u_64>();
            if (array3x3f != nullptr)
                return std::make_shared<Array3x3f>();
            if (array3x3d != nullptr)
                return std::make_shared<Array3x3d>();
            if (array3x3i_16 != nullptr)
                return std::make_shared<Array3x3i_16>();
            if (array3x3i != nullptr)
                return std::make_shared<Array3x3i>();
            if (array3x3i_64 != nullptr)
                return std::make_shared<Array3x3i_64>();
            if (array3x3u_16 != nullptr)
                return std::make_shared<Array3x3u_16>();
            if (array3x3u != nullptr)
                return std::make_shared<Array3x3u>();
            if (array3x3u_64 != nullptr)
                return std::make_shared<Array3x3u_64>();
            if (array4x4f != nullptr)
                return std::make_shared<Array4x4f>();
            if (array4x4d != nullptr)
                return std::make_shared<Array4x4d>();
            if (array4x4i_16 != nullptr)
                return std::make_shared<Array4x4i_16>();
            if (array4x4i != nullptr)
                return std::make_shared<Array4x4i>();
            if (array4x4i_64 != nullptr)
                return std::make_shared<Array4x4i_64>();
            if (array4x4u_16 != nullptr)
                return std::make_shared<Array4x4u_16>();
            if (array4x4u != nullptr)
                return std::make_shared<Array4x4u>();
            if (array4x4u_64 != nullptr)
                return std::make_shared<Array4x4u_64>();
            return nullptr;
        }

        /**
         * @brief Assigns a value from another devirtualized array into this devirtualized array.
         * @param ito Index in this array to write into.
         * @param ifrom Index of other array to read from.
         * @param other Other array to read from.
         */
        void assign(std::size_t ito, std::size_t ifrom, const DevirtualizedArray& other)
        {
            assert(0 <= ifrom && ifrom < other.iarray->getSize());
            assert(0 <= ito && ito < iarray->getSize());
            if (array1f != nullptr && other.array1f != nullptr)
                array1f->setValue(ito, other.array1f->getValue(ifrom));
            else if (array1d != nullptr && other.array1d != nullptr)
                array1d->setValue(ito, other.array1d->getValue(ifrom));
            else if (array1i_16 != nullptr && other.array1i_16 != nullptr)
                array1i_16->setValue(ito, other.array1i_16->getValue(ifrom));
            else if (array1i != nullptr && other.array1i != nullptr)
                array1i->setValue(ito, other.array1i->getValue(ifrom));
            else if (array1i_64 != nullptr && other.array1i_64 != nullptr)
                array1i_64->setValue(ito, other.array1i_64->getValue(ifrom));
            else if (array1u_16 != nullptr && other.array1u_16 != nullptr)
                array1u_16->setValue(ito, other.array1u_16->getValue(ifrom));
            else if (array1u != nullptr && other.array1u != nullptr)
                array1u->setValue(ito, other.array1u->getValue(ifrom));
            else if (array1u_64 != nullptr && other.array1u_64 != nullptr)
                array1u_64->setValue(ito, other.array1u_64->getValue(ifrom));
            else if (array2f != nullptr && other.array2f != nullptr)
                array2f->setValue(ito, other.array2f->getValue(ifrom));
            else if (array2d != nullptr && other.array2d != nullptr)
                array2d->setValue(ito, other.array2d->getValue(ifrom));
            else if (array2i_16 != nullptr && other.array2i_16 != nullptr)
                array2i_16->setValue(ito, other.array2i_16->getValue(ifrom));
            else if (array2i != nullptr && other.array2i != nullptr)
                array2i->setValue(ito, other.array2i->getValue(ifrom));
            else if (array2i_64 != nullptr && other.array2i_64 != nullptr)
                array2i_64->setValue(ito, other.array2i_64->getValue(ifrom));
            else if (array2u_16 != nullptr && other.array2u_16 != nullptr)
                array2u_16->setValue(ito, other.array2u_16->getValue(ifrom));
            else if (array2u != nullptr && other.array2u != nullptr)
                array2u->setValue(ito, other.array2u->getValue(ifrom));
            else if (array2u_64 != nullptr && other.array2u_64 != nullptr)
                array2u_64->setValue(ito, other.array2u_64->getValue(ifrom));
            else if (array3f != nullptr && other.array3f != nullptr)
                array3f->setValue(ito, other.array3f->getValue(ifrom));
            else if (array3d != nullptr && other.array3d != nullptr)
                array3d->setValue(ito, other.array3d->getValue(ifrom));
            else if (array3i_16 != nullptr && other.array3i_16 != nullptr)
                array3i_16->setValue(ito, other.array3i_16->getValue(ifrom));
            else if (array3i != nullptr && other.array3i != nullptr)
                array3i->setValue(ito, other.array3i->getValue(ifrom));
            else if (array3i_64 != nullptr && other.array3i_64 != nullptr)
                array3i_64->setValue(ito, other.array3i_64->getValue(ifrom));
            else if (array3u_16 != nullptr && other.array3u_16 != nullptr)
                array3u_16->setValue(ito, other.array3u_16->getValue(ifrom));
            else if (array3u != nullptr && other.array3u != nullptr)
                array3u->setValue(ito, other.array3u->getValue(ifrom));
            else if (array3u_64 != nullptr && other.array3u_64 != nullptr)
                array3u_64->setValue(ito, other.array3u_64->getValue(ifrom));
            else if (array4f != nullptr && other.array4f != nullptr)
                array4f->setValue(ito, other.array4f->getValue(ifrom));
            else if (array4d != nullptr && other.array4d != nullptr)
                array4d->setValue(ito, other.array4d->getValue(ifrom));
            else if (array4i_16 != nullptr && other.array4i_16 != nullptr)
                array4i_16->setValue(ito, other.array4i_16->getValue(ifrom));
            else if (array4i != nullptr && other.array4i != nullptr)
                array4i->setValue(ito, other.array4i->getValue(ifrom));
            else if (array4i_64 != nullptr && other.array4i_64 != nullptr)
                array4i_64->setValue(ito, other.array4i_64->getValue(ifrom));
            else if (array4u_16 != nullptr && other.array4u_16 != nullptr)
                array4u_16->setValue(ito, other.array4u_16->getValue(ifrom));
            else if (array4u != nullptr && other.array4u != nullptr)
                array4u->setValue(ito, other.array4u->getValue(ifrom));
            else if (array4u_64 != nullptr && other.array4u_64 != nullptr)
                array4u_64->setValue(ito, other.array4u_64->getValue(ifrom));
            else if (array2x2f != nullptr && other.array2x2f != nullptr)
                array2x2f->setValue(ito, other.array2x2f->getValue(ifrom));
            else if (array2x2d != nullptr && other.array2x2d != nullptr)
                array2x2d->setValue(ito, other.array2x2d->getValue(ifrom));
            else if (array2x2i_16 != nullptr && other.array2x2i_16 != nullptr)
                array2x2i_16->setValue(ito, other.array2x2i_16->getValue(ifrom));
            else if (array2x2i != nullptr && other.array2x2i != nullptr)
                array2x2i->setValue(ito, other.array2x2i->getValue(ifrom));
            else if (array2x2i_64 != nullptr && other.array2x2i_64 != nullptr)
                array2x2i_64->setValue(ito, other.array2x2i_64->getValue(ifrom));
            else if (array2x2u_16 != nullptr && other.array2x2u_16 != nullptr)
                array2x2u_16->setValue(ito, other.array2x2u_16->getValue(ifrom));
            else if (array2x2u != nullptr && other.array2x2u != nullptr)
                array2x2u->setValue(ito, other.array2x2u->getValue(ifrom));
            else if (array2x2u_64 != nullptr && other.array2x2u_64 != nullptr)
                array2x2u_64->setValue(ito, other.array2x2u_64->getValue(ifrom));
            else if (array3x3f != nullptr && other.array3x3f != nullptr)
                array3x3f->setValue(ito, other.array3x3f->getValue(ifrom));
            else if (array3x3d != nullptr && other.array3x3d != nullptr)
                array3x3d->setValue(ito, other.array3x3d->getValue(ifrom));
            else if (array3x3i_16 != nullptr && other.array3x3i_16 != nullptr)
                array3x3i_16->setValue(ito, other.array3x3i_16->getValue(ifrom));
            else if (array3x3i != nullptr && other.array3x3i != nullptr)
                array3x3i->setValue(ito, other.array3x3i->getValue(ifrom));
            else if (array3x3i_64 != nullptr && other.array3x3i_64 != nullptr)
                array3x3i_64->setValue(ito, other.array3x3i_64->getValue(ifrom));
            else if (array3x3u_16 != nullptr && other.array3x3u_16 != nullptr)
                array3x3u_16->setValue(ito, other.array3x3u_16->getValue(ifrom));
            else if (array3x3u != nullptr && other.array3x3u != nullptr)
                array3x3u->setValue(ito, other.array3x3u->getValue(ifrom));
            else if (array3x3u_64 != nullptr && other.array3x3u_64 != nullptr)
                array3x3u_64->setValue(ito, other.array3x3u_64->getValue(ifrom));
            else if (array4x4f != nullptr && other.array4x4f != nullptr)
                array4x4f->setValue(ito, other.array4x4f->getValue(ifrom));
            else if (array4x4d != nullptr && other.array4x4d != nullptr)
                array4x4d->setValue(ito, other.array4x4d->getValue(ifrom));
            else if (array4x4i_16 != nullptr && other.array4x4i_16 != nullptr)
                array4x4i_16->setValue(ito, other.array4x4i_16->getValue(ifrom));
            else if (array4x4i != nullptr && other.array4x4i != nullptr)
                array4x4i->setValue(ito, other.array4x4i->getValue(ifrom));
            else if (array4x4i_64 != nullptr && other.array4x4i_64 != nullptr)
                array4x4i_64->setValue(ito, other.array4x4i_64->getValue(ifrom));
            else if (array4x4u_16 != nullptr && other.array4x4u_16 != nullptr)
                array4x4u_16->setValue(ito, other.array4x4u_16->getValue(ifrom));
            else if (array4x4u != nullptr && other.array4x4u != nullptr)
                array4x4u->setValue(ito, other.array4x4u->getValue(ifrom));
            else if (array4x4u_64 != nullptr && other.array4x4u_64 != nullptr)
                array4x4u_64->setValue(ito, other.array4x4u_64->getValue(ifrom));
        }

        std::shared_ptr<IArray> iarray;
        std::shared_ptr<Array1f> array1f;
        std::shared_ptr<Array1d> array1d;
        std::shared_ptr<Array1i_16> array1i_16;
        std::shared_ptr<Array1i> array1i;
        std::shared_ptr<Array1i_64> array1i_64;
        std::shared_ptr<Array1u_16> array1u_16;
        std::shared_ptr<Array1u> array1u;
        std::shared_ptr<Array1u_64> array1u_64;
        std::shared_ptr<Array2f> array2f;
        std::shared_ptr<Array2d> array2d;
        std::shared_ptr<Array2i_16> array2i_16;
        std::shared_ptr<Array2i> array2i;
        std::shared_ptr<Array2i_64> array2i_64;
        std::shared_ptr<Array2u_16> array2u_16;
        std::shared_ptr<Array2u> array2u;
        std::shared_ptr<Array2u_64> array2u_64;
        std::shared_ptr<Array3f> array3f;
        std::shared_ptr<Array3d> array3d;
        std::shared_ptr<Array3i_16> array3i_16;
        std::shared_ptr<Array3i> array3i;
        std::shared_ptr<Array3i_64> array3i_64;
        std::shared_ptr<Array3u_16> array3u_16;
        std::shared_ptr<Array3u> array3u;
        std::shared_ptr<Array3u_64> array3u_64;
        std::shared_ptr<Array4f> array4f;
        std::shared_ptr<Array4d> array4d;
        std::shared_ptr<Array4i_16> array4i_16;
        std::shared_ptr<Array4i> array4i;
        std::shared_ptr<Array4i_64> array4i_64;
        std::shared_ptr<Array4u_16> array4u_16;
        std::shared_ptr<Array4u> array4u;
        std::shared_ptr<Array4u_64> array4u_64;
        std::shared_ptr<Array2x2f> array2x2f;
        std::shared_ptr<Array2x2d> array2x2d;
        std::shared_ptr<Array2x2i_16> array2x2i_16;
        std::shared_ptr<Array2x2i> array2x2i;
        std::shared_ptr<Array2x2i_64> array2x2i_64;
        std::shared_ptr<Array2x2u_16> array2x2u_16;
        std::shared_ptr<Array2x2u> array2x2u;
        std::shared_ptr<Array2x2u_64> array2x2u_64;
        std::shared_ptr<Array3x3f> array3x3f;
        std::shared_ptr<Array3x3d> array3x3d;
        std::shared_ptr<Array3x3i_16> array3x3i_16;
        std::shared_ptr<Array3x3i> array3x3i;
        std::shared_ptr<Array3x3i_64> array3x3i_64;
        std::shared_ptr<Array3x3u_16> array3x3u_16;
        std::shared_ptr<Array3x3u> array3x3u;
        std::shared_ptr<Array3x3u_64> array3x3u_64;
        std::shared_ptr<Array4x4f> array4x4f;
        std::shared_ptr<Array4x4d> array4x4d;
        std::shared_ptr<Array4x4i_16> array4x4i_16;
        std::shared_ptr<Array4x4i> array4x4i;
        std::shared_ptr<Array4x4i_64> array4x4i_64;
        std::shared_ptr<Array4x4u_16> array4x4u_16;
        std::shared_ptr<Array4x4u> array4x4u;
        std::shared_ptr<Array4x4u_64> array4x4u_64;
    };

    UpdateInfo FaceNormals3f::internalUpdate(ProgressInfo& progress)
    {
        // get input surface and output surface
        auto inSurfaces  = inputSurfaces.getData();
        auto outSurfaces = outputSurfaces.getData();

        // for each input surface
        for (std::size_t iSurface = 0; iSurface < inSurfaces->getNumSurfaces(); ++iSurface)
        {
            // get the input surface
            auto inSurface = inSurfaces->getSurface(iSurface);

            // is there an index buffer?
            if (inSurface->positions->getSize() > 0 && inSurface->indices->getSize() == 0)
            {
                return UpdateInfo::reportWarning("Surface " + std::to_string(iSurface) + " with positions, but without indices.");
            }

            // switch depending on primitive topology
            switch (inSurface->primitiveTopology)
            {
            case EPrimitiveTopology::TriangleList:
            {
                // create output surface
                auto outSurface = outSurfaces->createSurface();

                // get number of triangles
                Eigen::Index numTriangles = inSurface->indices->getSize() / 3;

                // allocate buffers for output
                outSurface->indices->setSize(numTriangles * 3);
                outSurface->positions->setSize(numTriangles * 3);
                outSurface->normals = std::make_shared<Array3f>();
                outSurface->normals->setSize(numTriangles * 3);
                if (inSurface->texCoords)
                    outSurface->texCoords->setSize(numTriangles * 3);
                outSurface->attributes->setSize(inSurface->attributes->getSize());

                // devirtualize the attributes and allocate output
                std::vector<std::shared_ptr<DevirtualizedArray>> inAttributes(inSurface->attributes->getSize());
                std::vector<std::shared_ptr<DevirtualizedArray>> outAttributes(inSurface->attributes->getSize());
                for (std::size_t iattr = 0; iattr < inSurface->attributes->getSize(); ++iattr)
                {
                    inAttributes[iattr]  = std::make_shared<DevirtualizedArray>(inSurface->attributes->getByIndex(iattr));
                    outAttributes[iattr] = std::make_shared<DevirtualizedArray>(inAttributes[iattr]->allocate());
                    outAttributes[iattr]->iarray->setSize(numTriangles * 3);
                    outSurface->attributes->setByIndex(iattr, outAttributes[iattr]->iarray);
                }

                // for each triangle
                for (Eigen::Index iTriangle = 0; iTriangle < numTriangles; ++iTriangle)
                {
                    // get input indices of the corners
                    uint32_t i0 = inSurface->indices->getValue(iTriangle * 3 + 0).x();
                    uint32_t i1 = inSurface->indices->getValue(iTriangle * 3 + 1).x();
                    uint32_t i2 = inSurface->indices->getValue(iTriangle * 3 + 2).x();

                    // get positions of the corners
                    Eigen::Vector3f p0 = inSurface->positions->getValue(i0);
                    Eigen::Vector3f p1 = inSurface->positions->getValue(i1);
                    Eigen::Vector3f p2 = inSurface->positions->getValue(i2);

                    // compute face normal
                    Eigen::Vector3f normal = (p1 - p0).cross(p2 - p0).stableNormalized();

                    // write the linear indices
                    outSurface->indices->setValue(iTriangle * 3 + 0, iTriangle * 3 + 0);
                    outSurface->indices->setValue(iTriangle * 3 + 1, iTriangle * 3 + 1);
                    outSurface->indices->setValue(iTriangle * 3 + 2, iTriangle * 3 + 2);

                    // write the positions
                    outSurface->positions->setValue(iTriangle * 3 + 0, p0);
                    outSurface->positions->setValue(iTriangle * 3 + 1, p1);
                    outSurface->positions->setValue(iTriangle * 3 + 2, p2);

                    // write the normals
                    outSurface->normals->setValue(iTriangle * 3 + 0, normal);
                    outSurface->normals->setValue(iTriangle * 3 + 1, normal);
                    outSurface->normals->setValue(iTriangle * 3 + 2, normal);

                    // write the texcoords
                    if (inSurface->texCoords)
                    {
                        // get texture coordinates of the corners
                        Eigen::Vector2f tx0 = inSurface->texCoords->getValue(i0);
                        Eigen::Vector2f tx1 = inSurface->texCoords->getValue(i1);
                        Eigen::Vector2f tx2 = inSurface->texCoords->getValue(i2);

                        // write the texture coordinates
                        outSurface->texCoords->setValue(iTriangle * 3 + 0, tx0);
                        outSurface->texCoords->setValue(iTriangle * 3 + 1, tx1);
                        outSurface->texCoords->setValue(iTriangle * 3 + 2, tx2);
                    }

                    // write the attributes
                    for (std::size_t iattr = 0; iattr < outSurface->attributes->getSize(); ++iattr)
                    {
                        outAttributes[iattr]->assign(iTriangle * 3 + 0, i0, *inAttributes[iattr]);
                        outAttributes[iattr]->assign(iTriangle * 3 + 1, i1, *inAttributes[iattr]);
                        outAttributes[iattr]->assign(iTriangle * 3 + 2, i2, *inAttributes[iattr]);
                    }
                }
                break;
            }
            default:
                return UpdateInfo::reportError("Algorithm not implemented for primitive topology.");
            }
        }

        // compute the new bounding box
        outSurfaces->recomputeBoundingBox();

        progress.allJobsDone();
        return UpdateInfo::reportValid();
    }
}
