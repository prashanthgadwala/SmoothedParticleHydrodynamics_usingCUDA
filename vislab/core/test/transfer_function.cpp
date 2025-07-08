#include "vislab/core/transfer_function.hpp"

#include "init_vislab.hpp"

#include "gtest/gtest.h"

namespace vislab
{
    template<typename TScalar>
    static void test_tf1d()
    {
        // create a transfer function
        auto tf = std::make_unique<TransferFunction<1, TScalar>>();
        tf->minValue = 0;
        tf->maxValue = 3;
        tf->values.insert(std::make_pair(0, Eigen::Vector<TScalar, 1>((TScalar)0.4)));
        tf->values.insert(std::make_pair(1, Eigen::Vector<TScalar, 1>((TScalar)0.6)));

        // exact interpolation at end points
        EXPECT_NEAR(tf->map(0).x(), 0.4, 1E-6);
        EXPECT_NEAR(tf->map(3).x(), 0.6, 1E-6);

        // interpolate in-between
        EXPECT_NEAR(tf->map(1.5).x(), 0.5, 1E-6);

        // clamp outside
        EXPECT_NEAR(tf->map(-1).x(), 0.4, 1E-6);
        EXPECT_NEAR(tf->map(4).x(), 0.6, 1E-6);
    }

    TEST(core, transfer_function)
    {
        Init();

        EXPECT_TRUE(Factory::create(TransferFunction1f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction1d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction2f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction2d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction3f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction3d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction4f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction4d::type()) != nullptr);

        test_tf1d<float>();
        test_tf1d<double>();
    }
}
