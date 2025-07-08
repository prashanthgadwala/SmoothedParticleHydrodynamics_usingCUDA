#pragma once

#include "surfaces.hpp"

#include <vislab/core/algorithm.hpp>

namespace vislab
{
    /**
     * @brief Algorithm that calculates face normals.
     * @details: The index buffer is linearized, such that vertices are duplicated for every face. This process loses the attributes of the surface.
     */
    class FaceNormals3f : public ConcreteAlgorithm<FaceNormals3f>
    {
    public:
        /**
         * @brief Constructor.
         */
        FaceNormals3f() = default;

        /**
         * @brief Constructor.
         * @param inSurfaces Surfaces that the positions are read from.
         * @param outSurfaces Surface that the face normals are written to.
         */
        FaceNormals3f(const std::shared_ptr<Surfaces3f>& inSurfaces, const std::shared_ptr<Surfaces3f>& outSurfaces)
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

    protected:
        /**
         * @brief Internal computation function
         * @param[in] progress Optional progress info.
         * @return Information about the completion of the computation, including a potential error message.
         */
        [[nodiscard]] UpdateInfo internalUpdate(ProgressInfo& progress) override;
    };
}
