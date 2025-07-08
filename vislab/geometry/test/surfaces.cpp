#include "vislab/geometry/surfaces.hpp"
#include "vislab/geometry/attributes.hpp"

#include "init_vislab.hpp"

#include "vislab/core/array.hpp"

#include "Eigen/Eigen"
#include "gtest/gtest.h"

namespace vislab
{
    TEST(geometry, surfaces)
    {
        Init();

        EXPECT_TRUE(Factory::create(Surface1f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surface1d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surface2f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surface2d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surface3f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surface3d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surface4f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surface4d::type()) != nullptr);

        EXPECT_TRUE(Factory::create(Surfaces1f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surfaces1d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surfaces2f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surfaces2d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surfaces3f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surfaces3d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surfaces4f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Surfaces4d::type()) != nullptr);

        auto surfacesA  = std::make_unique<Surfaces3f>();
        EXPECT_TRUE(surfacesA->isValid());

        auto surfacesA1 = surfacesA->createSurface();
        EXPECT_TRUE(surfacesA1->isValid());
        EXPECT_TRUE(surfacesA->isValid());

        surfacesA1->primitiveTopology = EPrimitiveTopology::TriangleList;
        surfacesA1->positions->append(Eigen::Vector3f(1.0f, 2.0f, 3.2f));
        surfacesA1->positions->append(Eigen::Vector3f(1.1f, 2.1f, 3.1f));
        surfacesA1->positions->append(Eigen::Vector3f(1.2f, 2.2f, 3.0f));
        EXPECT_FALSE(surfacesA1->isValid());
        EXPECT_FALSE(surfacesA->isValid());
        surfacesA1->recomputeBoundingBox();
        EXPECT_FALSE(surfacesA1->isValid());
        EXPECT_FALSE(surfacesA->isValid());
        surfacesA1->indices->append(Eigen::Vector1u(0));
        surfacesA1->indices->append(Eigen::Vector1u(1));
        surfacesA1->indices->append(Eigen::Vector1u(2));
        EXPECT_TRUE(surfacesA1->isValid());
        EXPECT_FALSE(surfacesA->isValid());
        surfacesA->recomputeBoundingBox();
        EXPECT_TRUE(surfacesA->isValid());
        EXPECT_EQ(surfacesA1->getBoundingBox().min().x(), 1.0f);
        EXPECT_EQ(surfacesA1->getBoundingBox().min().y(), 2.0f);
        EXPECT_EQ(surfacesA1->getBoundingBox().min().z(), 3.0f);
        EXPECT_EQ(surfacesA1->getBoundingBox().max().x(), 1.2f);
        EXPECT_EQ(surfacesA1->getBoundingBox().max().y(), 2.2f);
        EXPECT_EQ(surfacesA1->getBoundingBox().max().z(), 3.2f);
        auto surfaceA1Float = surfacesA1->attributes->create<Array1f>("float");
        surfaceA1Float->setSize(3);
        surfaceA1Float->setValue(0, 6.0f);
        surfaceA1Float->setValue(1, 6.1f);
        surfaceA1Float->setValue(2, 6.2f);

        auto surfacesA2 = surfacesA->createSurface();
        surfacesA2->primitiveTopology = EPrimitiveTopology::TriangleList;
        surfacesA2->positions->append(Eigen::Vector3f(3.0f, 4.0f, 5.0f));
        surfacesA2->positions->append(Eigen::Vector3f(3.1f, 4.1f, 5.2f));
        surfacesA2->positions->append(Eigen::Vector3f(3.2f, 4.2f, 5.1f));
        surfacesA2->recomputeBoundingBox();
        surfacesA2->indices->append(Eigen::Vector1u(0));
        surfacesA2->indices->append(Eigen::Vector1u(1));
        surfacesA2->indices->append(Eigen::Vector1u(2));
        EXPECT_EQ(surfacesA2->getBoundingBox().min().x(), 3.0f);
        EXPECT_EQ(surfacesA2->getBoundingBox().min().y(), 4.0f);
        EXPECT_EQ(surfacesA2->getBoundingBox().min().z(), 5.0f);
        EXPECT_EQ(surfacesA2->getBoundingBox().max().x(), 3.2f);
        EXPECT_EQ(surfacesA2->getBoundingBox().max().y(), 4.2f);
        EXPECT_EQ(surfacesA2->getBoundingBox().max().z(), 5.2f);
        auto surfaceA2Float = surfacesA2->attributes->create<Array1f>("float");
        surfaceA2Float->setSize(3);
        surfaceA2Float->setValue(0, 7.0f);
        surfaceA2Float->setValue(1, 7.1f);
        surfaceA2Float->setValue(2, 7.2f);

        auto surfacesB = surfacesA->clone();
        EXPECT_TRUE(surfacesB->isEqual(surfacesA.get()));
        EXPECT_FALSE(surfaceA1Float->isEqual(surfaceA2Float.get()));
    }
}
