#include "vislab/geometry/attributes.hpp"
#include "vislab/geometry/vertex_normals.hpp"
#include "vislab/geometry/surfaces.hpp"

#include "init_vislab.hpp"

#include "vislab/core/array.hpp"

#include "Eigen/Eigen"
#include "gtest/gtest.h"

namespace vislab
{
    TEST(geometry, vertex_normals)
    {
        Init();

        EXPECT_TRUE(Factory::create(VertexNormals3f::type()) != nullptr);

        auto surfaces = std::make_shared<Surfaces3f>();
        auto surface  = surfaces->createSurface();
        // add vertices
        surface->positions->append(Eigen::Vector3f(1.0f, 0.0f, 0.0f));
        surface->positions->append(Eigen::Vector3f(0.0f, 1.0f, 0.0f));
        surface->positions->append(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
        surface->positions->append(Eigen::Vector3f(0.0f, -1.0f, 0.0f));
        surfaces->recomputeBoundingBox();

        // add indices (triangle list topology)
        surface->primitiveTopology = EPrimitiveTopology::TriangleList;
        // triangle 1
        surface->indices->append(Eigen::Vector1u(0));
        surface->indices->append(Eigen::Vector1u(1));
        surface->indices->append(Eigen::Vector1u(2));
        // triangle 2
        surface->indices->append(Eigen::Vector1u(3));
        surface->indices->append(Eigen::Vector1u(0));
        surface->indices->append(Eigen::Vector1u(2));

        // add an attribute
        auto attr = surface->attributes->create<Array1d>("test");
        attr->append(Eigen::Vector1d(0.1));
        attr->append(Eigen::Vector1d(0.2));
        attr->append(Eigen::Vector1d(0.3));
        attr->append(Eigen::Vector1d(0.4));

        // compute face normals
        auto vertexNormal = std::make_unique<VertexNormals3f>();
        EXPECT_EQ(vertexNormal->getInputPorts().size(), 1);
        EXPECT_EQ(vertexNormal->getOutputPorts().size(), 1);
        EXPECT_EQ(vertexNormal->getParameters().size(), 0);

        vertexNormal->inputSurfaces.setData(surfaces);
        vertexNormal->outputSurfaces.setData(std::make_shared<Surfaces3f>());
        auto updateInfo = vertexNormal->update();
        EXPECT_TRUE(updateInfo.success());
        auto out_surfaces = vertexNormal->outputSurfaces.getData();

        // is there a surface on the output?
        EXPECT_TRUE(out_surfaces != nullptr);

        // correct number of surfaces?
        EXPECT_EQ(out_surfaces->getNumSurfaces(), 1);

        // correct bounding box?
        EXPECT_EQ(out_surfaces->getBoundingBox().min(), surfaces->getBoundingBox().min());
        EXPECT_EQ(out_surfaces->getBoundingBox().max(), surfaces->getBoundingBox().max());

        // get first surface
        auto out_surface = out_surfaces->getSurface(0);
        EXPECT_TRUE(out_surface != nullptr);

        // have buffers correct size?
        EXPECT_EQ(out_surface->positions->getSize(), 4);
        EXPECT_EQ(out_surface->indices->getSize(), 6);
        EXPECT_EQ(out_surface->attributes->getSize(), 1);
        EXPECT_EQ(out_surface->attributes->getByIndex(0)->getSize(), 4);
        EXPECT_EQ(out_surface->normals->getSize(), 4);

        // positions correct?
        EXPECT_EQ(out_surface->positions->getValue(0), surface->positions->getValue(0));
        EXPECT_EQ(out_surface->positions->getValue(1), surface->positions->getValue(1));
        EXPECT_EQ(out_surface->positions->getValue(2), surface->positions->getValue(2));
        EXPECT_EQ(out_surface->positions->getValue(3), surface->positions->getValue(3));

        // indices correct?
        EXPECT_EQ(out_surface->indices->getValue(0), Eigen::Vector1u(0));
        EXPECT_EQ(out_surface->indices->getValue(1), Eigen::Vector1u(1));
        EXPECT_EQ(out_surface->indices->getValue(2), Eigen::Vector1u(2));
        EXPECT_EQ(out_surface->indices->getValue(3), Eigen::Vector1u(3));
        EXPECT_EQ(out_surface->indices->getValue(4), Eigen::Vector1u(0));
        EXPECT_EQ(out_surface->indices->getValue(5), Eigen::Vector1u(2));

        // attribute correct?
        auto arr0 = std::dynamic_pointer_cast<Array1d>(out_surface->attributes->getByIndex(0));
        EXPECT_EQ(arr0->getValue(0), Eigen::Vector1d(0.1));
        EXPECT_EQ(arr0->getValue(1), Eigen::Vector1d(0.2));
        EXPECT_EQ(arr0->getValue(2), Eigen::Vector1d(0.3));
        EXPECT_EQ(arr0->getValue(3), Eigen::Vector1d(0.4));

        // normals correct?
        Eigen::Vector3f gt_normal0 = Eigen::Vector3f(1, 1, 1).stableNormalized();
        Eigen::Vector3f gt_normal1 = Eigen::Vector3f(1, -1, 1).stableNormalized();
        Eigen::Vector3f gt_normal2 = Eigen::Vector3f(1, 0, 1).stableNormalized();
        EXPECT_EQ(out_surface->normals->getValue(0), gt_normal2);
        EXPECT_EQ(out_surface->normals->getValue(1), gt_normal0);
        EXPECT_EQ(out_surface->normals->getValue(2), gt_normal2);
        EXPECT_EQ(out_surface->normals->getValue(3), gt_normal1);
    }
}
