#pragma once

#include <Eigen/Eigen>

namespace Eigen
{
    using Vector1i  = Vector<int32_t, 1>;
    using Vector1u  = Vector<uint32_t, 1>;
    using Vector1l  = Vector<int64_t, 1>;
    using Vector1f  = Vector<float, 1>;
    using Vector1d  = Vector<double, 1>;
    using Vector1cf = Vector<std::complex<float>, 1>;
    using Vector1cd = Vector<std::complex<double>, 1>;

    using VectorXl = Vector<int64_t, -1>;
    using Vector2l = Vector<int64_t, 2>;
    using Vector3l = Vector<int64_t, 3>;
    using Vector4l = Vector<int64_t, 4>;

    using VectorXu = Vector<uint32_t, -1>;
    using Vector2u = Vector<uint32_t, 2>;
    using Vector3u = Vector<uint32_t, 3>;
    using Vector4u = Vector<uint32_t, 4>;

    using MatrixXl = Matrix<int64_t, -1, -1>;
    using Matrix2l = Matrix<int64_t, 2, 2>;
    using Matrix3l = Matrix<int64_t, 3, 3>;
    using Matrix4l = Matrix<int64_t, 4, 4>;

    using MatrixXu = Matrix<uint32_t, -1, -1>;
    using Matrix2u = Matrix<uint32_t, 2, 2>;
    using Matrix3u = Matrix<uint32_t, 3, 3>;
    using Matrix4u = Matrix<uint32_t, 4, 4>;
}
