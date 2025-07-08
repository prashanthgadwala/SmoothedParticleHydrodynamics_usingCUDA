#include "vislab/graphics/image.hpp"
#include "init_vislab.hpp"

#include "Eigen/Eigen"
#include "gtest/gtest.h"

namespace vislab
{
    /**
     * @brief Test function for images.
     * @tparam TArray Array type to run the tests for.
     */
    template <typename TImage>
    void test_image()
    {
        // allocate an image. initially all set to zero?
        auto image = std::make_unique<TImage>();
        EXPECT_EQ(image->getResolution().x(), 0);
        EXPECT_EQ(image->getResolution().y(), 0);
        EXPECT_EQ(image->getNumPixels(), 0);

        // set the resolution. is array resized?
        image->setResolution(3, 5);
        EXPECT_EQ(image->getResolution().x(), 3);
        EXPECT_EQ(image->getResolution().y(), 5);
        EXPECT_EQ(image->getNumPixels(), 15);
        EXPECT_EQ(image->getArray()->getSize(), 15);

        // clear the array
        image->setZero();
        const int channels = TImage::Channels;
        for (int c = 0; c < channels; ++c)
            EXPECT_EQ(image->getValue(0, 1)[c], 0);

        // set a value at a specific index. value readable?
        typename TImage::Element element;
        for (int c = 0; c < channels; ++c)
            element[c] = c;
        image->setValue(0, 1, element);
        for (int c = 0; c < channels; ++c)
        {
            EXPECT_EQ(image->getValue(0, 1)[c], c);
        }
    }

    TEST(graphics, image)
    {
        Init();

        test_image<Image1f>();
        test_image<Image2f>();
        test_image<Image3f>();
        test_image<Image4f>();
        test_image<Image1d>();
        test_image<Image2d>();
        test_image<Image3d>();
        test_image<Image4d>();
    }
}
