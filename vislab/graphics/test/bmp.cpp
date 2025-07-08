#include "init_vislab.hpp"
#include "vislab/graphics/bmp_reader.hpp"
#include "vislab/graphics/bmp_writer.hpp"

#include "Eigen/Eigen"
#include "gtest/gtest.h"

namespace vislab
{
    TEST(graphics, bmp)
    {
        Init();

        // generate a test image
        auto imageOut = std::make_shared<vislab::Image3d>();
        int width     = 5;
        int height    = 6;
        imageOut->setResolution(width, height);
        imageOut->setZero();
        for (int iw = 0; iw < width; ++iw)
            for (int ih = 0; ih < height; ++ih)
            {
                bool on = (iw + ih) % 2 == 0;
                imageOut->setValue(iw, ih,
                                   on ? Eigen::Vector3d(1, 0, 0) : Eigen::Vector3d(0, 0, 1));
            }

        // write the image to a file
        auto writer = std::make_unique<vislab::BmpWriter>();
        writer->paramPath.setValue("img.bmp");
        writer->inputImage.setData(imageOut);
        EXPECT_TRUE(writer->update().success());

        // number of ports and parameters correct?
        EXPECT_EQ(writer->getInputPorts().size(), 1);
        EXPECT_EQ(writer->getOutputPorts().size(), 0);
        EXPECT_EQ(writer->getParameters().size(), 1);

        // read the image from a file
        auto reader = std::make_unique<vislab::BmpReader>();
        reader->paramPath.setValue("img.bmp");
        reader->outputImage.setData(std::make_shared<Image3d>());
        EXPECT_TRUE(reader->update().success());
        auto imageIn = reader->outputImage.getData();

        // number of ports and parameters correct?
        EXPECT_EQ(reader->getInputPorts().size(), 0);
        EXPECT_EQ(reader->getOutputPorts().size(), 1);
        EXPECT_EQ(reader->getParameters().size(), 1);

        // read image correct?
        EXPECT_EQ(imageIn->getResolution().x(), 5);
        EXPECT_EQ(imageIn->getResolution().y(), 6);
        for (int iw = 0; iw < width; ++iw)
            for (int ih = 0; ih < height; ++ih)
            {
                EXPECT_EQ(imageIn->getValue(iw, ih).x(), imageOut->getValue(iw, ih).x());
                EXPECT_EQ(imageIn->getValue(iw, ih).y(), imageOut->getValue(iw, ih).y());
                EXPECT_EQ(imageIn->getValue(iw, ih).z(), imageOut->getValue(iw, ih).z());
            }
    }
}
