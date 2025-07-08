#pragma once

#include "surfaces.hpp"

#include <vislab/core/algorithm.hpp>

namespace vislab
{
    /**
     * @brief Algorithm that calculates vertex normals.
     */
    class VertexNormals3f : public ConcreteAlgorithm<VertexNormals3f>
    {
    public:
        /**
         * @brief Constructor.
         */
        VertexNormals3f() = default;

        /**
         * @brief Constructor.
         * @param inSurfaces Surfaces that the positions are read from.
         * @param outSurfaces Surface that the face normals are written to.
         */
        VertexNormals3f(const std::shared_ptr<Surfaces3f>& inSurfaces, const std::shared_ptr<Surfaces3f>& outSurfaces)
        {
            inputSurfaces.setData(inSurfaces);
            outputSurfaces.setData(outSurfaces);
        }


        /**
         * @brief Surfaces that the positions are read from.
         */
        InputPort<Surfaces3f> inputSurfaces;

        /**
         * @brief Surface that the face normals are written to.
         */
        OutputPort<Surfaces3f> outputSurfaces;

        /**
         * @brief Computes vertex normals and adds them to the given surface.
         * @param positions Vertex positions of the surface.
         * @param indices Index buffer of the surface.
         * @param normals Buffer that stores the resulting normals.
        */
        static void computeNormals(std::shared_ptr<const Array3f> positions, std::shared_ptr<const Array1u> indices, std::shared_ptr<Array3f> normals);

        /**
         * @brief Computes vertex normals and adds them to the given surface.
         * @param positions Vertex positions of the surface.
         * @param indices Index buffer of the surface.
         * @param normals Buffer that stores the resulting normals.
         */
        static void computeNormals(std::shared_ptr<const Array3f> positions, std::shared_ptr<const Array3u> indices, std::shared_ptr<Array3f> normals);

        /**
         * @brief Computes vertex normals and adds them to the given surface.
         * @param surfaces Surface to compute the vertex normals for.
         */
        static void computeNormals(std::shared_ptr<Surface3f> surface);

        /**
         * @brief Computes vertex normals and adds them to the given set of surfaces.
         * @param surfaces Surfaces to compute the vertex normals for.
        */
        static void computeNormals(std::shared_ptr<Surfaces3f> surfaces);

    protected:
        /**
         * @brief Internal computation function
         * @param[in] progress Optional progress info.
         * @return Information about the completion of the computation, including a potential error message.
         */
        [[nodiscard]] UpdateInfo internalUpdate(ProgressInfo& progress) override;
    };
}
