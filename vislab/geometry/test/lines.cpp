#include "vislab/geometry/lines.hpp"
#include "vislab/geometry/attributes.hpp"

#include "init_vislab.hpp"

#include "vislab/core/array.hpp"

#include "Eigen/Eigen"
#include "gtest/gtest.h"

namespace vislab
{
    TEST(geometry, lines)
    {
        Init();

        EXPECT_TRUE(Factory::create(Line1f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Line1d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Line2f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Line2d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Line3f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Line3d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Line4f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Line4d::type()) != nullptr);

        EXPECT_TRUE(Factory::create(Lines1f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Lines1d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Lines2f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Lines2d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Lines3f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Lines3d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Lines4f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Lines4d::type()) != nullptr);

        auto linesA  = std::make_unique<Lines2f>();
        EXPECT_TRUE(linesA->isValid());

        auto linesA1 = linesA->createLine();
        EXPECT_TRUE(linesA1->isValid());
        EXPECT_TRUE(linesA->isValid());

        linesA1->vertices->append(Eigen::Vector2f(1.0f, 2.0f));
        linesA1->vertices->append(Eigen::Vector2f(1.1f, 2.1f));
        linesA1->vertices->append(Eigen::Vector2f(1.2f, 2.2f));
        EXPECT_FALSE(linesA1->isValid());
        EXPECT_FALSE(linesA->isValid());
        linesA1->recomputeBoundingBox();
        EXPECT_TRUE(linesA1->isValid());
        EXPECT_FALSE(linesA->isValid());
        linesA->recomputeBoundingBox();
        EXPECT_TRUE(linesA->isValid());

        auto lineA1Float = linesA1->attributes->create<Array1f>("float");
        lineA1Float->setSize(3);
        lineA1Float->setValue(0, 6.0f);
        lineA1Float->setValue(1, 6.1f);
        lineA1Float->setValue(2, 6.2f);
        EXPECT_NEAR(linesA1->arcLength(), sqrtf(0.08f), 1E-6);

        auto linesA2 = linesA->createLine();
        linesA2->vertices->append(Eigen::Vector2f(3.0f, 4.0f));
        linesA2->vertices->append(Eigen::Vector2f(3.1f, 4.1f));
        linesA2->vertices->append(Eigen::Vector2f(3.2f, 4.2f));
        linesA2->recomputeBoundingBox();
        auto lineA2Float = linesA2->attributes->create<Array1f>("float");
        lineA2Float->setSize(3);
        lineA2Float->setValue(0, 7.0f);
        lineA2Float->setValue(1, 7.1f);
        lineA2Float->setValue(2, 7.2f);
        EXPECT_NEAR(linesA2->arcLength(), sqrtf(0.08f), 1E-6);

        auto linesB = linesA->clone();
        EXPECT_TRUE(linesB->isEqual(linesA.get()));
    }
}
