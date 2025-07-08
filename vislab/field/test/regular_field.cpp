#include <vislab/core/array.hpp>
#include <vislab/field/regular_field.hpp>

#include "init_vislab.hpp"

#include "Eigen/Eigen"
#include "random"
#include "gtest/gtest.h"

namespace vislab
{
    TEST(field, regular)
    {
        Init();

        EXPECT_TRUE(Factory::create(RegularGrid1d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularGrid2d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularGrid3d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularGrid4d::type()) != nullptr);

        EXPECT_TRUE(Factory::create(RegularSteadyScalarField2f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularSteadyScalarField2d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularSteadyScalarField3f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularSteadyScalarField3d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularUnsteadyScalarField2f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularUnsteadyScalarField2d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularUnsteadyScalarField3f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularUnsteadyScalarField3d::type()) != nullptr);

        EXPECT_TRUE(Factory::create(RegularSteadyVectorField2f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularSteadyVectorField2d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularSteadyVectorField3f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularSteadyVectorField3d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularUnsteadyVectorField2f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularUnsteadyVectorField2d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularUnsteadyVectorField3f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularUnsteadyVectorField3d::type()) != nullptr);

        EXPECT_TRUE(Factory::create(RegularSteadyTensorField2f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularSteadyTensorField2d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularSteadyTensorField3f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularSteadyTensorField3d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularUnsteadyTensorField2f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularUnsteadyTensorField2d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularUnsteadyTensorField3f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(RegularUnsteadyTensorField3d::type()) != nullptr);

        // create a test grid
        auto grid = std::make_shared<RegularGrid3d>();
        grid->setDomain(Eigen::AlignedBox3d(Eigen::Vector3d(0.1, 0.2, 0.3), Eigen::Vector3d(1.1, 1.2, 1.3)));
        grid->setResolution(Eigen::Vector3i(16, 16, 16));

        // create a cubic scalar test data at grid coordinates
        auto array = std::make_shared<Array1d>();
        array->setSize(grid->getResolution().prod());
        for (int iz = 0; iz < grid->getResolution().z(); ++iz)
            for (int iy = 0; iy < grid->getResolution().y(); ++iy)
                for (int ix = 0; ix < grid->getResolution().x(); ++ix)
                {
                    Eigen::Index linearIndex = grid->getLinearIndex({ ix, iy, iz });
                    Eigen::Vector3d coord    = grid->getCoordAt(linearIndex);
                    double x = coord.x(), y = coord.y(), z = coord.z();
                    double s = 2 * x * x * x + x * x * y + 2 * x * y * y + 3 * y * y * y - 2 * y + x + x * x * z + 3 * z * z * y - 2 * z * z * z + z * 2;
                    array->setValue(linearIndex, s);
                }

        // place data and grid in field data structure
        auto field = std::make_shared<RegularSteadyScalarField3d>();
        field->setGrid(grid);
        field->setArray(array);
        field->setAccuracy(3); // third-order accurate reconstruction
        field->setBoundaryBehavior(0, EBoundaryBehavior::Clamp);
        field->setBoundaryBehavior(1, EBoundaryBehavior::Clamp);
        field->setBoundaryBehavior(2, EBoundaryBehavior::Clamp);

        for (int iz = 0; iz < grid->getResolution().z(); ++iz)
            for (int iy = 0; iy < grid->getResolution().y(); ++iy)
                for (int ix = 0; ix < grid->getResolution().x(); ++ix)
                {
                    Eigen::Index linearIndex = grid->getLinearIndex({ ix, iy, iz });
                    Eigen::Vector3d coord    = grid->getCoordAt(linearIndex);
                    double x = coord.x(), y = coord.y(), z = coord.z();
                    double s   = 2 * x * x * x + x * x * y + 2 * x * y * y + 3 * y * y * y - 2 * y + x + x * x * z + 3 * z * z * y - 2 * z * z * z + z * 2;
                    double sx  = 2 * x * z + 2 * y * y + 2 * x * y + 6 * x * x + 1;
                    double sy  = 3 * z * z + 9 * y * y + 4 * x * y + x * x - 2;
                    double sz  = -6 * z * z + 6 * y * z + x * x + 2;
                    double sxx = 2 * z + 2 * y + 12 * x;
                    double sxy = 4 * y + 2 * x;
                    double sxz = 2 * x;
                    double syy = 18 * y + 4 * x;
                    double syz = 6 * z;
                    double szz = 6 * y - 12 * z;

                    EXPECT_NEAR(s, field->sample(coord).x(), 1E-8);
                    EXPECT_NEAR(sx, field->samplePartial(coord, { 1, 0, 0 }).x(), 1E-8);
                    EXPECT_NEAR(sy, field->samplePartial(coord, { 0, 1, 0 }).x(), 1E-8);
                    EXPECT_NEAR(sz, field->samplePartial(coord, { 0, 0, 1 }).x(), 1E-8);
                    EXPECT_NEAR(sxx, field->samplePartial(coord, { 2, 0, 0 }).x(), 1E-8);
                    EXPECT_NEAR(sxy, field->samplePartial(coord, { 1, 1, 0 }).x(), 1E-8);
                    EXPECT_NEAR(sxz, field->samplePartial(coord, { 1, 0, 1 }).x(), 1E-8);
                    EXPECT_NEAR(syy, field->samplePartial(coord, { 0, 2, 0 }).x(), 1E-8);
                    EXPECT_NEAR(syz, field->samplePartial(coord, { 0, 1, 1 }).x(), 1E-8);
                    EXPECT_NEAR(szz, field->samplePartial(coord, { 0, 0, 2 }).x(), 1E-8);
                }
    }

    TEST(field, regular_partial)
    {
        // allocate grid
        auto grid = std::make_shared<RegularGrid3d>();
        grid->setResolution(Eigen::Vector3i(5, 6, 7));
        grid->setDomain(Eigen::AlignedBox3d(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1.1, 1.3, 1.6)));

        // put random values in grid
        std::default_random_engine rng;
        std::uniform_real_distribution<double> rnd;
        auto arr = std::make_shared<Array1d>();
        arr->setSize(grid->getResolution().prod());
        for (Eigen::Index i = 0; i < arr->getSize(); ++i)
            arr->setValue(i, rnd(rng));

        // define field
        RegularSteadyScalarField3d sfield;
        sfield.setGrid(grid);
        sfield.setArray(arr);

        // compute partial at each point
        for (Eigen::Index i = 0; i < grid->getResolution().prod(); ++i)
        {
            sfield.setAccuracy(2);

            auto gridCoord = grid->getGridCoord(i);

            // first derivatives
            EXPECT_NEAR(
                sfield.estimatePartial(gridCoord, RegularSteadyScalarField3d::Partial::dx).x(),
                sfield.samplePartial(sfield.getGrid()->getCoordAt(gridCoord), RegularSteadyScalarField3d::Partial::dx).x(),
                1E-8);

            EXPECT_NEAR(
                sfield.estimatePartial(gridCoord, RegularSteadyScalarField3d::Partial::dy).x(),
                sfield.samplePartial(sfield.getGrid()->getCoordAt(gridCoord), RegularSteadyScalarField3d::Partial::dy).x(),
                1E-8);

            EXPECT_NEAR(
                sfield.estimatePartial(gridCoord, RegularSteadyScalarField3d::Partial::dz).x(),
                sfield.samplePartial(sfield.getGrid()->getCoordAt(gridCoord), RegularSteadyScalarField3d::Partial::dz).x(),
                1E-8);

            // second derivatives
            EXPECT_NEAR(
                sfield.estimatePartial(gridCoord, RegularSteadyScalarField3d::Partial::dxx).x(),
                sfield.samplePartial(sfield.getGrid()->getCoordAt(gridCoord), RegularSteadyScalarField3d::Partial::dxx).x(),
                1E-8);

            EXPECT_NEAR(
                sfield.estimatePartial(gridCoord, RegularSteadyScalarField3d::Partial::dyy).x(),
                sfield.samplePartial(sfield.getGrid()->getCoordAt(gridCoord), RegularSteadyScalarField3d::Partial::dyy).x(),
                1E-8);

            EXPECT_NEAR(
                sfield.estimatePartial(gridCoord, RegularSteadyScalarField3d::Partial::dzz).x(),
                sfield.samplePartial(sfield.getGrid()->getCoordAt(gridCoord), RegularSteadyScalarField3d::Partial::dzz).x(),
                1E-8);
        }
    }
}
