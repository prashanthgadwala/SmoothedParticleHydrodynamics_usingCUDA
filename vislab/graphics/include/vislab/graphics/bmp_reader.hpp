#pragma once

#include "image.hpp"

#include <vislab/core/algorithm.hpp>
#include <vislab/core/path_parameter.hpp>

namespace vislab
{
    /**
     * @brief Bitmap reader for RGB images.
     */
    class BmpReader : public ConcreteAlgorithm<BmpReader>
    {
    public:
        /**
         * @brief Constructor.
         * @param path Path to read from.
         */
        BmpReader(const std::string& path = "")
            : paramPath(PathParameter::EFile::In, "Bitmap (*.bmp)")
        {
            paramPath.setValue(path);
        }

        /**
         * @brief Constructor.
         * @param outImage Image to write to.
         * @param path Path to read from.
         */
        BmpReader(const std::shared_ptr<Image3d>& outImage, const std::string& path = "")
            : BmpReader(path)
        {
            outputImage.setData(outImage);
        }

        /**
         * @brief Image data that was read from file.
         */
        OutputPort<Image3d> outputImage;

        /**
         * @brief Path to read from.
         */
        PathParameter paramPath;

    protected:
        /**
         * @brief Internal computation function
         * @param progress Optional progress info.
         * @return Information about the completion of the computation, including a potential error message.
         */
        [[nodiscard]] UpdateInfo internalUpdate(ProgressInfo& progress) override;
    };
}
