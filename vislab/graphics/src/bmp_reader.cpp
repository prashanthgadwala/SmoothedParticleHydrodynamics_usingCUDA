#include <vislab/graphics/bmp_reader.hpp>

#include <fstream>

namespace vislab
{
    // ==========================================================================================

    UpdateInfo BmpReader::internalUpdate(ProgressInfo& progress)
    {
        // get output
        auto image = outputImage.getData();

        // get the path parameter
        const std::string& path = paramPath.getValue();

        // open the file
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp)
            return UpdateInfo::reportError("File could not be opened!");

        // reader the header information
        int width, height;
        short bit_count; // bit count per pixel (24 if no palette)
        fseek(fp, 18, SEEK_SET);
        fread(&width, sizeof(int), 1, fp);
        fread(&height, sizeof(int), 1, fp);
        fseek(fp, 28, SEEK_SET);
        fread(&bit_count, sizeof(short), 1, fp);
        fseek(fp, 54, SEEK_SET);

        // read image data as unsigned char array in [0,255]
        int width_file      = (width + 3) / 4 * 4; // Increase the width to make sure it's multiple of 4
        int size            = width_file * height;
        unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * size * 3);
        if (!data)
            return UpdateInfo::reportError("File too large to allocate!");
        if (bit_count == 24)
            fread(data, sizeof(unsigned char), (std::size_t)size * 3, fp);
        else
            return UpdateInfo::reportError("Bit count is not 24!");
        fclose(fp);

        // copy into a Image3D object
        image->setResolution(width, height);
        int padSize = (4 - (width * 3) % 4) % 4;
        for (int iy = 0; iy < height; ++iy)
        {
            for (int ix = 0; ix < width; ++ix)
            {
                int linIndex = (iy * width + ix) * 3 + iy * padSize;
                Eigen::Vector3d color(
                    data[linIndex + 2] / 255.,
                    data[linIndex + 1] / 255.,
                    data[linIndex + 0] / 255.);
                image->setValue(ix, iy, color);
            }
        }
        free(data);

        progress.allJobsDone();
        return UpdateInfo::reportValid();
    }
}
