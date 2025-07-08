#include <vislab/field/base_partial.hpp>

#include "gtest/gtest.h"

namespace vislab
{
    TEST(field, base_partial)
    {
        EXPECT_EQ(PartialSteady2d(0, 0).hash, PartialSteady2d::c);
        EXPECT_EQ(PartialSteady2d(1, 0).hash, PartialSteady2d::dx);
        EXPECT_EQ(PartialSteady2d(0, 1).hash, PartialSteady2d::dy);
        EXPECT_EQ(PartialSteady2d(2, 0).hash, PartialSteady2d::dxx);
        EXPECT_EQ(PartialSteady2d(1, 1).hash, PartialSteady2d::dxy);
        EXPECT_EQ(PartialSteady2d(0, 2).hash, PartialSteady2d::dyy);
        EXPECT_EQ(PartialSteady2d(3, 0).hash, PartialSteady2d::dxxx);
        EXPECT_EQ(PartialSteady2d(2, 1).hash, PartialSteady2d::dxxy);
        EXPECT_EQ(PartialSteady2d(1, 2).hash, PartialSteady2d::dxyy);
        EXPECT_EQ(PartialSteady2d(0, 3).hash, PartialSteady2d::dyyy);

        EXPECT_EQ(PartialUnsteady2d(0, 0, 0).hash, PartialUnsteady2d::c);
        EXPECT_EQ(PartialUnsteady2d(1, 0, 0).hash, PartialUnsteady2d::dx);
        EXPECT_EQ(PartialUnsteady2d(0, 1, 0).hash, PartialUnsteady2d::dy);
        EXPECT_EQ(PartialUnsteady2d(0, 0, 1).hash, PartialUnsteady2d::dt);
        EXPECT_EQ(PartialUnsteady2d(2, 0, 0).hash, PartialUnsteady2d::dxx);
        EXPECT_EQ(PartialUnsteady2d(1, 1, 0).hash, PartialUnsteady2d::dxy);
        EXPECT_EQ(PartialUnsteady2d(1, 0, 1).hash, PartialUnsteady2d::dxt);
        EXPECT_EQ(PartialUnsteady2d(0, 2, 0).hash, PartialUnsteady2d::dyy);
        EXPECT_EQ(PartialUnsteady2d(0, 1, 1).hash, PartialUnsteady2d::dyt);
        EXPECT_EQ(PartialUnsteady2d(0, 0, 2).hash, PartialUnsteady2d::dtt);
        EXPECT_EQ(PartialUnsteady2d(3, 0, 0).hash, PartialUnsteady2d::dxxx);
        EXPECT_EQ(PartialUnsteady2d(2, 1, 0).hash, PartialUnsteady2d::dxxy);
        EXPECT_EQ(PartialUnsteady2d(2, 0, 1).hash, PartialUnsteady2d::dxxt);
        EXPECT_EQ(PartialUnsteady2d(1, 2, 0).hash, PartialUnsteady2d::dxyy);
        EXPECT_EQ(PartialUnsteady2d(1, 1, 1).hash, PartialUnsteady2d::dxyt);
        EXPECT_EQ(PartialUnsteady2d(1, 0, 2).hash, PartialUnsteady2d::dxtt);
        EXPECT_EQ(PartialUnsteady2d(0, 3, 0).hash, PartialUnsteady2d::dyyy);
        EXPECT_EQ(PartialUnsteady2d(0, 2, 1).hash, PartialUnsteady2d::dyyt);
        EXPECT_EQ(PartialUnsteady2d(0, 1, 2).hash, PartialUnsteady2d::dytt);
        EXPECT_EQ(PartialUnsteady2d(0, 0, 3).hash, PartialUnsteady2d::dttt);

        EXPECT_EQ(PartialSteady3d(0, 0, 0).hash, PartialSteady3d::c);
        EXPECT_EQ(PartialSteady3d(1, 0, 0).hash, PartialSteady3d::dx);
        EXPECT_EQ(PartialSteady3d(0, 1, 0).hash, PartialSteady3d::dy);
        EXPECT_EQ(PartialSteady3d(0, 0, 1).hash, PartialSteady3d::dz);
        EXPECT_EQ(PartialSteady3d(2, 0, 0).hash, PartialSteady3d::dxx);
        EXPECT_EQ(PartialSteady3d(1, 1, 0).hash, PartialSteady3d::dxy);
        EXPECT_EQ(PartialSteady3d(1, 0, 1).hash, PartialSteady3d::dxz);
        EXPECT_EQ(PartialSteady3d(0, 2, 0).hash, PartialSteady3d::dyy);
        EXPECT_EQ(PartialSteady3d(0, 1, 1).hash, PartialSteady3d::dyz);
        EXPECT_EQ(PartialSteady3d(0, 0, 2).hash, PartialSteady3d::dzz);
        EXPECT_EQ(PartialSteady3d(3, 0, 0).hash, PartialSteady3d::dxxx);
        EXPECT_EQ(PartialSteady3d(2, 1, 0).hash, PartialSteady3d::dxxy);
        EXPECT_EQ(PartialSteady3d(2, 0, 1).hash, PartialSteady3d::dxxz);
        EXPECT_EQ(PartialSteady3d(1, 2, 0).hash, PartialSteady3d::dxyy);
        EXPECT_EQ(PartialSteady3d(1, 1, 1).hash, PartialSteady3d::dxyz);
        EXPECT_EQ(PartialSteady3d(1, 0, 2).hash, PartialSteady3d::dxzz);
        EXPECT_EQ(PartialSteady3d(0, 3, 0).hash, PartialSteady3d::dyyy);
        EXPECT_EQ(PartialSteady3d(0, 2, 1).hash, PartialSteady3d::dyyz);
        EXPECT_EQ(PartialSteady3d(0, 1, 2).hash, PartialSteady3d::dyzz);
        EXPECT_EQ(PartialSteady3d(0, 0, 3).hash, PartialSteady3d::dzzz);

        EXPECT_EQ(PartialUnsteady3d(0, 0, 0, 0).hash, PartialUnsteady3d::c);
        EXPECT_EQ(PartialUnsteady3d(1, 0, 0, 0).hash, PartialUnsteady3d::dx);
        EXPECT_EQ(PartialUnsteady3d(0, 1, 0, 0).hash, PartialUnsteady3d::dy);
        EXPECT_EQ(PartialUnsteady3d(0, 0, 1, 0).hash, PartialUnsteady3d::dz);
        EXPECT_EQ(PartialUnsteady3d(0, 0, 0, 1).hash, PartialUnsteady3d::dt);
        EXPECT_EQ(PartialUnsteady3d(2, 0, 0, 0).hash, PartialUnsteady3d::dxx);
        EXPECT_EQ(PartialUnsteady3d(1, 1, 0, 0).hash, PartialUnsteady3d::dxy);
        EXPECT_EQ(PartialUnsteady3d(1, 0, 1, 0).hash, PartialUnsteady3d::dxz);
        EXPECT_EQ(PartialUnsteady3d(1, 0, 0, 1).hash, PartialUnsteady3d::dxt);
        EXPECT_EQ(PartialUnsteady3d(0, 2, 0, 0).hash, PartialUnsteady3d::dyy);
        EXPECT_EQ(PartialUnsteady3d(0, 1, 1, 0).hash, PartialUnsteady3d::dyz);
        EXPECT_EQ(PartialUnsteady3d(0, 1, 0, 1).hash, PartialUnsteady3d::dyt);
        EXPECT_EQ(PartialUnsteady3d(0, 0, 2, 0).hash, PartialUnsteady3d::dzz);
        EXPECT_EQ(PartialUnsteady3d(0, 0, 1, 1).hash, PartialUnsteady3d::dzt);
        EXPECT_EQ(PartialUnsteady3d(0, 0, 0, 2).hash, PartialUnsteady3d::dtt);
        EXPECT_EQ(PartialUnsteady3d(3, 0, 0, 0).hash, PartialUnsteady3d::dxxx);
        EXPECT_EQ(PartialUnsteady3d(2, 1, 0, 0).hash, PartialUnsteady3d::dxxy);
        EXPECT_EQ(PartialUnsteady3d(2, 0, 1, 0).hash, PartialUnsteady3d::dxxz);
        EXPECT_EQ(PartialUnsteady3d(2, 0, 0, 1).hash, PartialUnsteady3d::dxxt);
        EXPECT_EQ(PartialUnsteady3d(1, 2, 0, 0).hash, PartialUnsteady3d::dxyy);
        EXPECT_EQ(PartialUnsteady3d(1, 1, 1, 0).hash, PartialUnsteady3d::dxyz);
        EXPECT_EQ(PartialUnsteady3d(1, 1, 0, 1).hash, PartialUnsteady3d::dxyt);
        EXPECT_EQ(PartialUnsteady3d(1, 0, 2, 0).hash, PartialUnsteady3d::dxzz);
        EXPECT_EQ(PartialUnsteady3d(1, 0, 1, 1).hash, PartialUnsteady3d::dxzt);
        EXPECT_EQ(PartialUnsteady3d(1, 0, 0, 2).hash, PartialUnsteady3d::dxtt);
        EXPECT_EQ(PartialUnsteady3d(0, 3, 0, 0).hash, PartialUnsteady3d::dyyy);
        EXPECT_EQ(PartialUnsteady3d(0, 2, 1, 0).hash, PartialUnsteady3d::dyyz);
        EXPECT_EQ(PartialUnsteady3d(0, 2, 0, 1).hash, PartialUnsteady3d::dyyt);
        EXPECT_EQ(PartialUnsteady3d(0, 1, 2, 0).hash, PartialUnsteady3d::dyzz);
        EXPECT_EQ(PartialUnsteady3d(0, 1, 1, 1).hash, PartialUnsteady3d::dyzt);
        EXPECT_EQ(PartialUnsteady3d(0, 1, 0, 2).hash, PartialUnsteady3d::dytt);
        EXPECT_EQ(PartialUnsteady3d(0, 0, 3, 0).hash, PartialUnsteady3d::dzzz);
        EXPECT_EQ(PartialUnsteady3d(0, 0, 2, 1).hash, PartialUnsteady3d::dzzt);
        EXPECT_EQ(PartialUnsteady3d(0, 0, 0, 3).hash, PartialUnsteady3d::dttt);
    }
}
