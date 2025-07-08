#pragma once

#include "base_transfer_function_fwd.hpp"

namespace vislab
{
    template <int64_t TComponents, typename TScalar>
    class TransferFunction;

    /**
     * @brief Transfer function that maps to Eigen::Vector1d.
     */
    using TransferFunction1d = TransferFunction<1, double>;

    /**
     * @brief Transfer function that maps to Eigen::Vector2d.
     */
    using TransferFunction2d = TransferFunction<2, double>;

    /**
     * @brief Transfer function that maps to Eigen::Vector3d.
     */
    using TransferFunction3d = TransferFunction<3, double>;

    /**
     * @brief Transfer function that maps to Eigen::Vector4d.
     */
    using TransferFunction4d = TransferFunction<4, double>;

    /**
     * @brief Transfer function that maps to Eigen::Vector1f.
     */
    using TransferFunction1f = TransferFunction<1, float>;

    /**
     * @brief Transfer function that maps to Eigen::Vector2f.
     */
    using TransferFunction2f = TransferFunction<2, float>;

    /**
     * @brief Transfer function that maps to Eigen::Vector3f.
     */
    using TransferFunction3f = TransferFunction<3, float>;

    /**
     * @brief Transfer function that maps to Eigen::Vector4f.
     */
    using TransferFunction4f = TransferFunction<4, float>;
}
