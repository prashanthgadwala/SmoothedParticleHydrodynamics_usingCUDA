#pragma once

#include "base_image.hpp"

#include <vislab/core/algorithm.hpp>
#include <vislab/core/path_parameter.hpp>

namespace vislab
{
    /**
     * @brief Bitmap writer for RGB images.
     */
    class BmpWriter : public ConcreteAlgorithm<BmpWriter>
    {
    public:
        /**
         * @brief Constructor.
         * @param path Path to write to.
         */
        BmpWriter(const std::string& path = "")
            : paramPath(PathParameter::EFile::Out, "Bitmap (*.bmp)")
        {
            paramPath.setValue(path);
        }

        /**
         * @brief Constructor.
         * @param inImage Image to write.
         * @param path Path to write to.
         */
        BmpWriter(const std::shared_ptr<IImage3>& inImage, const std::string& path = "")
            : BmpWriter(path)
        {
            inputImage.setData(inImage);
        }

        /**
         * @brief Image data to write to a bitmap file.
         */
        InputPort<IImage3> inputImage;

        /**
         * @brief Path to write to.
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
