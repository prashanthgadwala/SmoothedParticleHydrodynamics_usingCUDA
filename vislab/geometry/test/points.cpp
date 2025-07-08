#include "vislab/geometry/points.hpp"
#include "vislab/geometry/attributes.hpp"

#include "init_vislab.hpp"

#include "vislab/core/array.hpp"

#include "Eigen/Eigen"
#include "gtest/gtest.h"

namespace vislab
{
    TEST(geometry, points)
    {
        Init();

        EXPECT_TRUE(Factory::create(Points1f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Points1d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Points2f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Points2d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Points3f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Points3d::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Points4f::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Points4d::type()) != nullptr);

        auto pointsA = std::make_unique<Points2f>();
        EXPECT_TRUE(pointsA->isValid());
        pointsA->getVertices()->append(Eigen::Vector2f(1.0f, 2.0f));
        pointsA->getVertices()->append(Eigen::Vector2f(1.1f, 2.1f));
        pointsA->getVertices()->append(Eigen::Vector2f(1.2f, 2.2f));
        EXPECT_FALSE(pointsA->isValid());
        pointsA->recomputeBoundingBox();
        EXPECT_TRUE(pointsA->isValid());

        auto pointsArev = std::make_unique<Points2f>();
        pointsArev->getVertices()->setValues({ Eigen::Vector2f(1.2f, 2.2f), Eigen::Vector2f(1.1f, 2.1f), Eigen::Vector2f(1.0f, 2.0f) });
        EXPECT_FALSE(pointsArev->isValid());
        pointsArev->recomputeBoundingBox();
        EXPECT_TRUE(pointsArev->isValid());

        auto pointsB = std::make_unique<Points2f>();
        pointsB->getVertices()->setValues({ Eigen::Vector2f(11.0f, 12.0f), Eigen::Vector2f(11.1f, 12.1f), Eigen::Vector2f(11.2f, 12.2f), Eigen::Vector2f(11.3f, 12.3f) });
        pointsB->recomputeBoundingBox();

        auto pointsAB_gt = std::make_unique<Points2f>();
        pointsAB_gt->getVertices()->setValues({ Eigen::Vector2f(1.0f, 2.0f), Eigen::Vector2f(1.1f, 2.1f), Eigen::Vector2f(1.2f, 2.2f), Eigen::Vector2f(11.0f, 12.0f), Eigen::Vector2f(11.1f, 12.1f), Eigen::Vector2f(11.2f, 12.2f), Eigen::Vector2f(11.3f, 12.3f) });
        pointsAB_gt->recomputeBoundingBox();

        auto pointsBA_gt = std::make_unique<Points2f>();
        pointsBA_gt->getVertices()->setValues({ Eigen::Vector2f(11.0f, 12.0f), Eigen::Vector2f(11.1f, 12.1f), Eigen::Vector2f(11.2f, 12.2f), Eigen::Vector2f(11.3f, 12.3f), Eigen::Vector2f(1.0f, 2.0f), Eigen::Vector2f(1.1f, 2.1f), Eigen::Vector2f(1.2f, 2.2f) });
        pointsBA_gt->recomputeBoundingBox();

        auto pointsTest = std::make_unique<Points2f>();
        pointsTest->getVertices()->setValues({ Eigen::Vector2f(1.0f, 2.0f), Eigen::Vector2f(1.1f, 2.1f), Eigen::Vector2f(1.2f, 2.2f) });
        pointsTest->recomputeBoundingBox();

        EXPECT_TRUE(pointsA->isEqual(pointsTest.get()));
        EXPECT_FALSE(pointsA->isEqual(pointsB.get()));

        pointsTest->append(pointsB.get());
        pointsTest->recomputeBoundingBox();
        EXPECT_TRUE(pointsTest->isEqual(pointsAB_gt.get()));

        pointsTest->removeLast(pointsB->getVertices()->getSize());
        pointsTest->recomputeBoundingBox();
        EXPECT_TRUE(pointsTest->isEqual(pointsA.get()));

        pointsTest->getVertices()->setValues({ Eigen::Vector2f(1.0f, 2.0f), Eigen::Vector2f(1.1f, 2.1f), Eigen::Vector2f(1.2f, 2.2f) });
        pointsTest->prepend(pointsB.get());
        pointsTest->recomputeBoundingBox();
        EXPECT_TRUE(pointsTest->isEqual(pointsBA_gt.get()));

        pointsTest->removeFirst(pointsB->getVertices()->getSize());
        pointsTest->recomputeBoundingBox();
        EXPECT_TRUE(pointsTest->isEqual(pointsA.get()));

        pointsTest->getVertices()->setValues({ Eigen::Vector2f(1.0f, 2.0f), Eigen::Vector2f(1.1f, 2.1f), Eigen::Vector2f(1.2f, 2.2f) });
        pointsTest->reverse();
        EXPECT_TRUE(pointsTest->isEqual(pointsArev.get()));

        auto pointsAattr = std::make_unique<Points2f>();
        pointsAattr->getVertices()->setValues({ Eigen::Vector2f(1.0f, 2.0f), Eigen::Vector2f(1.1f, 2.1f), Eigen::Vector2f(1.2f, 2.2f) });
        pointsAattr->recomputeBoundingBox();
        auto pointsAattr_float = pointsAattr->attributes->create<Array1f>("float");
        pointsAattr_float->setValues({ 30.f, 31.f, 32.f });
        auto pointsAattr_vec2f = pointsAattr->attributes->create<Array2f>("Vec2f");
        pointsAattr_vec2f->setValues({ Eigen::Vector2f(40.f, 50.f), Eigen::Vector2f(41.f, 51.f), Eigen::Vector2f(42.f, 52.f) });

        auto pointsAattrrev = std::make_unique<Points2f>();
        pointsAattrrev->getVertices()->setValues({ Eigen::Vector2f(1.2f, 2.2f), Eigen::Vector2f(1.1f, 2.1f), Eigen::Vector2f(1.0f, 2.0f) });
        pointsAattrrev->recomputeBoundingBox();
        auto pointsAattrrev_float = pointsAattrrev->attributes->create<Array1f>("float");
        pointsAattrrev_float->setValues({ 32.f, 31.f, 30.f, 13.f });
        EXPECT_FALSE(pointsAattrrev->isValid());
        pointsAattrrev_float->setValues({ 32.f, 31.f, 30.f });
        EXPECT_TRUE(pointsAattrrev->isValid());
        auto pointsAattrrev_vec2f = pointsAattrrev->attributes->create<Array2f>("Vec2f");
        pointsAattrrev_vec2f->setValues({ Eigen::Vector2f(42.f, 52.f), Eigen::Vector2f(41.f, 51.f), Eigen::Vector2f(40.f, 50.f) });

        auto pointsBattr = std::make_unique<Points2f>();
        pointsBattr->getVertices()->setValues({ Eigen::Vector2f(11.0f, 12.0f), Eigen::Vector2f(11.1f, 12.1f), Eigen::Vector2f(11.2f, 12.2f), Eigen::Vector2f(11.3f, 12.3f) });
        pointsBattr->recomputeBoundingBox();
        auto pointsBattr_float = pointsBattr->attributes->create<Array1f>("float");
        pointsBattr_float->setValues({ 130.f, 131.f, 132.f, 133.f });
        auto pointsBattr_vec2f = pointsBattr->attributes->create<Array2f>("Vec2f");
        pointsBattr_vec2f->setValues({ Eigen::Vector2f(140.f, 150.f), Eigen::Vector2f(141.f, 151.f), Eigen::Vector2f(142.f, 152.f), Eigen::Vector2f(143.f, 153.f) });

        auto pointsAB_gtattr = std::make_unique<Points2f>();
        pointsAB_gtattr->getVertices()->setValues({ Eigen::Vector2f(1.0f, 2.0f), Eigen::Vector2f(1.1f, 2.1f), Eigen::Vector2f(1.2f, 2.2f), Eigen::Vector2f(11.0f, 12.0f), Eigen::Vector2f(11.1f, 12.1f), Eigen::Vector2f(11.2f, 12.2f), Eigen::Vector2f(11.3f, 12.3f) });
        pointsAB_gtattr->recomputeBoundingBox();
        auto pointsAB_gtattr_float = pointsAB_gtattr->attributes->create<Array1f>("float");
        pointsAB_gtattr_float->setValues({ 30.f, 31.f, 32.f, 130.f, 131.f, 132.f, 133.f });
        auto pointsAB_gtattr_vec2f = pointsAB_gtattr->attributes->create<Array2f>("Vec2f");
        pointsAB_gtattr_vec2f->setValues({ Eigen::Vector2f(40.f, 50.f), Eigen::Vector2f(41.f, 51.f), Eigen::Vector2f(42.f, 52.f), Eigen::Vector2f(140.f, 150.f), Eigen::Vector2f(141.f, 151.f), Eigen::Vector2f(142.f, 152.f), Eigen::Vector2f(143.f, 153.f) });

        auto pointsBA_gtattr = std::make_unique<Points2f>();
        pointsBA_gtattr->getVertices()->setValues({ Eigen::Vector2f(11.0f, 12.0f), Eigen::Vector2f(11.1f, 12.1f), Eigen::Vector2f(11.2f, 12.2f), Eigen::Vector2f(11.3f, 12.3f), Eigen::Vector2f(1.0f, 2.0f), Eigen::Vector2f(1.1f, 2.1f), Eigen::Vector2f(1.2f, 2.2f) });
        pointsBA_gtattr->recomputeBoundingBox();
        auto pointsBA_gtattr_float = pointsBA_gtattr->attributes->create<Array1f>("float");
        pointsBA_gtattr_float->setValues({ 130.f, 131.f, 132.f, 133.f, 30.f, 31.f, 32.f });
        auto pointsBA_gtattr_vec2f = pointsBA_gtattr->attributes->create<Array2f>("Vec2f");
        pointsBA_gtattr_vec2f->setValues({ Eigen::Vector2f(140.f, 150.f), Eigen::Vector2f(141.f, 151.f), Eigen::Vector2f(142.f, 152.f), Eigen::Vector2f(143.f, 153.f), Eigen::Vector2f(40.f, 50.f), Eigen::Vector2f(41.f, 51.f), Eigen::Vector2f(42.f, 52.f) });

        pointsTest = std::make_unique<Points2f>();
        pointsTest->getVertices()->setValues({ Eigen::Vector2f(1.0f, 2.0f), Eigen::Vector2f(1.1f, 2.1f), Eigen::Vector2f(1.2f, 2.2f) });
        pointsTest->recomputeBoundingBox();
        auto pointsAattr_float_ = pointsTest->attributes->create<Array1f>("float");
        pointsAattr_float_->setValues({ 30.f, 31.f, 32.f });
        auto pointsAattr_vec2f_ = pointsTest->attributes->create<Array2f>("Vec2f");
        pointsAattr_vec2f_->setValues({ Eigen::Vector2f(40.f, 50.f), Eigen::Vector2f(41.f, 51.f), Eigen::Vector2f(42.f, 52.f) });
        pointsTest->append(pointsBattr.get());
        pointsTest->recomputeBoundingBox();
        EXPECT_TRUE(pointsTest->isEqual(pointsAB_gtattr.get()));
        pointsTest->removeLast(pointsBattr->getVertices()->getSize());
        pointsTest->recomputeBoundingBox();
        EXPECT_TRUE(pointsTest->isEqual(pointsAattr.get()));

        pointsTest = std::make_unique<Points2f>();
        pointsTest->getVertices()->setValues({ Eigen::Vector2f(1.0f, 2.0f), Eigen::Vector2f(1.1f, 2.1f), Eigen::Vector2f(1.2f, 2.2f) });
        pointsTest->recomputeBoundingBox();
        auto pointsAattr_float__ = pointsTest->attributes->create<Array1f>("float");
        pointsAattr_float__->setValues({ 30.f, 31.f, 32.f });
        auto pointsAattr_vec2f__ = pointsTest->attributes->create<Array2f>("Vec2f");
        pointsAattr_vec2f__->setValues({ Eigen::Vector2f(40.f, 50.f), Eigen::Vector2f(41.f, 51.f), Eigen::Vector2f(42.f, 52.f) });
        pointsTest->prepend(pointsBattr.get());
        pointsTest->recomputeBoundingBox();
        EXPECT_TRUE(pointsTest->isEqual(pointsBA_gtattr.get()));
        pointsTest->removeFirst(pointsBattr->getVertices()->getSize());
        pointsTest->recomputeBoundingBox();
        EXPECT_TRUE(pointsTest->isEqual(pointsAattr.get()));

        pointsTest->reverse();
        EXPECT_TRUE(pointsTest->isEqual(pointsAattrrev.get()));
        pointsTest->reverse();

        pointsTest->attributes->clear();
        EXPECT_TRUE(pointsTest->isEqual(pointsA.get()));

        pointsTest->attributes = pointsAattr->attributes;
        EXPECT_TRUE(pointsTest->isEqual(pointsAattr.get()));

        auto cloned = pointsTest->clone();
        EXPECT_TRUE(cloned->isEqual(pointsTest.get()));
    }
}
