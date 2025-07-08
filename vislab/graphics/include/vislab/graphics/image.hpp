#pragma once

#include "image_fwd.hpp"

#include "base_image.hpp"

#include <vislab/core/array.hpp>
#include <vislab/core/iarchive.hpp>
#include <vislab/core/traits.hpp>

namespace vislab
{
    /**
     * @brief Class that stores 2D image data.
     * @tparam TArrayType Internal array type used to store the pixel values.
     */
    template <typename TArrayType>
    class Image : public Concrete<Image<TArrayType>, BaseImage<TArrayType::Dimensions>>
    {
    public:
        /**
         * @brief Internal array type used to store the pixels.
         */
        using ArrayType = TArrayType;

        /**
         * @brief Data type for a single pixel.
         */
        using Element = typename TArrayType::Element;

        /**
         * @brief Empty default constructor.
         */
        Image()
            : mResolution(Eigen::Vector2i(0, 0))
            , mData(nullptr)
        {
        }

        /**
         * @brief Constructor of image with a certain size.
         * @param resolution Image resolution.
         */
        Image(const Eigen::Vector2i& resolution)
            : mResolution(resolution)
        {
            mData = std::make_shared<TArrayType>();
            mData->setSize(resolution.x() * resolution.y());
        }

        /**
         * @brief Constructor of image with a certain size.
         * @param width Width of image.
         * @param height Height of image.
         */
        Image(int width, int height)
            : mResolution(Eigen::Vector2i(width, height))
        {
            mData = std::make_shared<TArrayType>();
            mData->setSize(width * height);
        }

        /**
         * @brief Destructor.
         */
        ~Image() {}

        /**
         * @brief Sets the value at a certain pixel.
         * @param px x-coordinate of pixel.
         * @param py y-coordinate of pixel.
         * @param value Value to set.
         */
        inline void setValue(int px, int py, const Element& value) { setValue((Eigen::Index)py * mResolution.x() + px, value); }

        /**
         * @brief Sets the value at a certain pixel.
         * @param pixel Pixel index.
         * @param value Value to set.
         */
        inline void setValue(const Eigen::Vector2i& pixel, const Element& value) { setValue(pixel.y() * mResolution.x() + pixel.x(), value); }

        /**
         * @brief Sets the value at a certain pixel.
         * @param linearIndex Linear array index.
         * @param value Value to set.
         */
        inline void setValue(Eigen::Index linearIndex, const Element& value) { mData->setValue(linearIndex, value); }

        /**
         * @brief Gets the value at a certain pixel.
         * @param px x-coordinate of pixel.
         * @param py y-coordinate of pixel.
         * @return Value at pixel.
         */
        [[nodiscard]] inline const typename TArrayType::Element& getValue(int px, int py) const { return getValue((Eigen::Index)py * mResolution.x() + px); }

        /**
         * @brief Gets the value at a certain pixel.
         * @param pixel Pixel index.
         * @return Value at pixel.
         */
        [[nodiscard]] inline const typename TArrayType::Element& getValue(const Eigen::Vector2i& pixel) const { return getValue(pixel.y() * mResolution.x() + pixel.x()); }

        /**
         * @brief Gets the value at a certain pixel.
         * @param linearIndex Linear array index.
         * @return Value at pixel.
         */
        [[nodiscard]] inline const typename TArrayType::Element& getValue(Eigen::Index linearIndex) const { return mData->getValue(linearIndex); }

        /**
         * @brief Gets the resolution of the image.
         * @return Image resolution.
         */
        [[nodiscard]] inline const Eigen::Vector2i& getResolution() const override { return mResolution; }

        /**
         * @brief Sets the resolution of the image. The internal linear image data array gets resized.
         * @param resolution New image resolution.
         */
        void setResolution(const Eigen::Vector2i& resolution) override
        {
            mResolution = resolution;
            if (mData == nullptr)
                mData = std::make_shared<TArrayType>();
            mData->setSize(resolution.x() * resolution.y());
        }

        /**
         * @brief Sets the resolution of the image.
         * @param resx New width of the image.
         * @param resy New height of the image.
         */
        inline void setResolution(int resx, int resy) override
        {
            setResolution(Eigen::Vector2i(resx, resy));
        }

        /**
         * @brief Gets the total number of pixels.
         * @return Number of pixels.
         */
        [[nodiscard]] inline Eigen::Index getNumPixels() const override { return mResolution.x() * mResolution.y(); }

        /**
         * @brief Sets all values to zero.
         */
        inline void setZero() override
        {
            if (mData != nullptr)
                mData->setZero();
        }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override
        {
            archive("Resolution", mResolution);
            archive("Data", mData);
        }

        /**
         * @brief Gets the array that stores the data in the image.
         * @return Internal data array.
         */
        [[nodiscard]] inline std::shared_ptr<TArrayType> getArray() { return mData; }

        /**
         * @brief Gets the array that stores the data in the image.
         * @return Internal data array.
         */
        [[nodiscard]] inline std::shared_ptr<const TArrayType> getArray() const { return mData; }

    private:
        /**
         * @brief Image resolution.
         */
        Eigen::Vector2i mResolution;

        /**
         * @brief Data stored in the image.
         */
        std::shared_ptr<TArrayType> mData;
    };
}
