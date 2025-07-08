#pragma once

#include <stdint.h>

#include "types.hpp"

namespace vislab
{
    /**
     * @brief Type-safe data array with one or multiple components.
     * @tparam TElement Element type that is stored in the array.
     */
    template <typename TElement>
    class Array;

    /**
     * @brief Dense one-dimensional array of float values with one component.
     */
    using Array1f = Array<Eigen::Vector1f>;

    /**
     * @brief Dense one-dimensional array of float values with two components.
     */
    using Array2f = Array<Eigen::Vector2f>;

    /**
     * @brief Dense one-dimensional array of float values with three components.
     */
    using Array3f = Array<Eigen::Vector3f>;

    /**
     * @brief Dense one-dimensional array of float values with four components.
     */
    using Array4f = Array<Eigen::Vector4f>;

    /**
     * @brief Dense one-dimensional array of float 2x2 matrices.
     */
    using Array2x2f = Array<Eigen::Matrix2f>;

    /**
     * @brief Dense one-dimensional array of float 3x3 matrices.
     */
    using Array3x3f = Array<Eigen::Matrix3f>;

    /**
     * @brief Dense one-dimensional array of float 4x4 matrices.
     */
    using Array4x4f = Array<Eigen::Matrix4f>;

    /**
     * @brief Dense one-dimensional array of double values with one component.
     */
    using Array1d = Array<Eigen::Vector1d>;

    /**
     * @brief Dense one-dimensional array of double values with two components.
     */
    using Array2d = Array<Eigen::Vector2d>;

    /**
     * @brief Dense one-dimensional array of double values with three components.
     */
    using Array3d = Array<Eigen::Vector3d>;

    /**
     * @brief Dense one-dimensional array of double values with four components.
     */
    using Array4d = Array<Eigen::Vector4d>;

    /**
     * @brief Dense one-dimensional array of double 2x2 matrices.
     */
    using Array2x2d = Array<Eigen::Matrix2d>;

    /**
     * @brief Dense one-dimensional array of double 3x3 matrices.
     */
    using Array3x3d = Array<Eigen::Matrix3d>;

    /**
     * @brief Dense one-dimensional array of double 4x4 matrices.
     */
    using Array4x4d = Array<Eigen::Matrix4d>;

    /**
     * @brief Dense one-dimensional array of int16_t values with one component.
     */
    using Array1i_16 = Array<Eigen::Vector<int16_t, 1>>;

    /**
     * @brief Dense one-dimensional array of int16_t values with two components.
     */
    using Array2i_16 = Array<Eigen::Vector<int16_t, 2>>;

    /**
     * @brief Dense one-dimensional array of int16_t values with three components.
     */
    using Array3i_16 = Array<Eigen::Vector<int16_t, 3>>;

    /**
     * @brief Dense one-dimensional array of int16_t values with four components.
     */
    using Array4i_16 = Array<Eigen::Vector<int16_t, 4>>;

    /**
     * @brief Dense one-dimensional array of int16_t 2x2 matrices.
     */
    using Array2x2i_16 = Array<Eigen::Matrix<int16_t, 2, 2>>;

    /**
     * @brief Dense one-dimensional array of int16_t 3x3 matrices.
     */
    using Array3x3i_16 = Array<Eigen::Matrix<int16_t, 3, 3>>;

    /**
     * @brief Dense one-dimensional array of int16_t 4x4 matrices.
     */
    using Array4x4i_16 = Array<Eigen::Matrix<int16_t, 4, 4>>;

    /**
     * @brief Dense one-dimensional array of int32_t values with one component.
     */
    using Array1i = Array<Eigen::Vector<int32_t, 1>>;

    /**
     * @brief Dense one-dimensional array of int32_t values with two components.
     */
    using Array2i = Array<Eigen::Vector<int32_t, 2>>;

    /**
     * @brief Dense one-dimensional array of int32_t values with three components.
     */
    using Array3i = Array<Eigen::Vector<int32_t, 3>>;

    /**
     * @brief Dense one-dimensional array of int32_t values with four components.
     */
    using Array4i = Array<Eigen::Vector<int32_t, 4>>;

    /**
     * @brief Dense one-dimensional array of int32_t 2x2 matrices.
     */
    using Array2x2i = Array<Eigen::Matrix<int32_t, 2, 2>>;

    /**
     * @brief Dense one-dimensional array of int32_t 3x3 matrices.
     */
    using Array3x3i = Array<Eigen::Matrix<int32_t, 3, 3>>;

    /**
     * @brief Dense one-dimensional array of int32_t 4x4 matrices.
     */
    using Array4x4i = Array<Eigen::Matrix<int32_t, 4, 4>>;

    /**
     * @brief Dense one-dimensional array of int64_t values with one component.
     */
    using Array1i_64 = Array<Eigen::Vector<int64_t, 1>>;

    /**
     * @brief Dense one-dimensional array of int64_t values with two components.
     */
    using Array2i_64 = Array<Eigen::Vector<int64_t, 2>>;

    /**
     * @brief Dense one-dimensional array of int64_t values with three components.
     */
    using Array3i_64 = Array<Eigen::Vector<int64_t, 3>>;

    /**
     * @brief Dense one-dimensional array of int64_t values with four components.
     */
    using Array4i_64 = Array<Eigen::Vector<int64_t, 4>>;

    /**
     * @brief Dense one-dimensional array of int64_t 2x2 matrices.
     */
    using Array2x2i_64 = Array<Eigen::Matrix<int64_t, 2, 2>>;

    /**
     * @brief Dense one-dimensional array of int64_t 3x3 matrices.
     */
    using Array3x3i_64 = Array<Eigen::Matrix<int64_t, 3, 3>>;

    /**
     * @brief Dense one-dimensional array of int64_t 4x4 matrices.
     */
    using Array4x4i_64 = Array<Eigen::Matrix<int64_t, 4, 4>>;

    /**
     * @brief Dense one-dimensional array of uint16_t values with one component.
     */
    using Array1u_16 = Array<Eigen::Vector<uint16_t, 1>>;

    /**
     * @brief Dense one-dimensional array of uint16_t values with two components.
     */
    using Array2u_16 = Array<Eigen::Vector<uint16_t, 2>>;

    /**
     * @brief Dense one-dimensional array of uint16_t values with three components.
     */
    using Array3u_16 = Array<Eigen::Vector<uint16_t, 3>>;

    /**
     * @brief Dense one-dimensional array of uint16_t values with four components.
     */
    using Array4u_16 = Array<Eigen::Vector<uint16_t, 4>>;

    /**
     * @brief Dense one-dimensional array of uint16_t 2x2 matrices.
     */
    using Array2x2u_16 = Array<Eigen::Matrix<uint16_t, 2, 2>>;

    /**
     * @brief Dense one-dimensional array of uint16_t 3x3 matrices.
     */
    using Array3x3u_16 = Array<Eigen::Matrix<uint16_t, 3, 3>>;

    /**
     * @brief Dense one-dimensional array of uint16_t 4x4 matrices.
     */
    using Array4x4u_16 = Array<Eigen::Matrix<uint16_t, 4, 4>>;

    /**
     * @brief Dense one-dimensional array of uint32_t values with one component.
     */
    using Array1u = Array<Eigen::Vector<uint32_t, 1>>;

    /**
     * @brief Dense one-dimensional array of uint32_t values with two components.
     */
    using Array2u = Array<Eigen::Vector<uint32_t, 2>>;

    /**
     * @brief Dense one-dimensional array of uint32_t values with three components.
     */
    using Array3u = Array<Eigen::Vector<uint32_t, 3>>;

    /**
     * @brief Dense one-dimensional array of uint32_t values with four components.
     */
    using Array4u = Array<Eigen::Vector<uint32_t, 4>>;

    /**
     * @brief Dense one-dimensional array of uint32_t 2x2 matrices.
     */
    using Array2x2u = Array<Eigen::Matrix<uint32_t, 2, 2>>;

    /**
     * @brief Dense one-dimensional array of uint32_t 3x3 matrices.
     */
    using Array3x3u = Array<Eigen::Matrix<uint32_t, 3, 3>>;

    /**
     * @brief Dense one-dimensional array of uint32_t 4x4 matrices.
     */
    using Array4x4u = Array<Eigen::Matrix<uint32_t, 4, 4>>;

    /**
     * @brief Dense one-dimensional array of uint64_t values with one component.
     */
    using Array1u_64 = Array<Eigen::Vector<uint64_t, 1>>;

    /**
     * @brief Dense one-dimensional array of uint64_t values with two components.
     */
    using Array2u_64 = Array<Eigen::Vector<uint64_t, 2>>;

    /**
     * @brief Dense one-dimensional array of uint64_t values with three components.
     */
    using Array3u_64 = Array<Eigen::Vector<uint64_t, 3>>;

    /**
     * @brief Dense one-dimensional array of uint64_t values with four components.
     */
    using Array4u_64 = Array<Eigen::Vector<uint64_t, 4>>;

    /**
     * @brief Dense one-dimensional array of uint64_t 2x2 matrices.
     */
    using Array2x2u_64 = Array<Eigen::Matrix<uint64_t, 2, 2>>;

    /**
     * @brief Dense one-dimensional array of uint64_t 3x3 matrices.
     */
    using Array3x3u_64 = Array<Eigen::Matrix<uint64_t, 3, 3>>;

    /**
     * @brief Dense one-dimensional array of uint64_t 4x4 matrices.
     */
    using Array4x4u_64 = Array<Eigen::Matrix<uint64_t, 4, 4>>;

}
