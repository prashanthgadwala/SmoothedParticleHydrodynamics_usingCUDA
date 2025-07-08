#include "vislab/graphics/transform.hpp"

#include "Eigen/Eigen"
#include "gtest/gtest.h"

namespace vislab
{
    TEST(graphics, transform)
    {
        Eigen::Matrix4d A = Eigen::Matrix4d::Identity();
        A(3, 3)           = 2;
        
        Transform t;
        t.setMatrix(A);

        EXPECT_NEAR((t.transformPoint(Eigen::Vector3d(2, 4, 6)) - Eigen::Vector3d(1, 2, 3)).squaredNorm(), 0, 1E-7);

        A(1, 1) = .5;
        t.setMatrix(A);

        EXPECT_NEAR((t.transformVector(Eigen::Vector3d(2, 4, 6)) - Eigen::Vector3d(2, 2, 6)).squaredNorm(), 0, 1E-7);
    }
}
