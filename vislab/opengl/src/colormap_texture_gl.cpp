#include <vislab/opengl/colormap_texture_gl.hpp>

#include <vislab/opengl/transfer_function_gl.hpp>

#include <vislab/field/regular_field.hpp>

namespace vislab
{
    ColormapTextureGl::ColormapTextureGl(std::shared_ptr<const ColormapTexture> colormapTexture)
        : Concrete<ColormapTextureGl, ResourceGl<ColormapTexture>>(colormapTexture)
        , mTexture(0)
        , mTransferFunction(colormapTexture->transferFunction)
    {
    }

    ColormapTextureGl::~ColormapTextureGl()
    {
        this->releaseDevice();
    }

    void ColormapTextureGl::update(SceneGl* scene, OpenGL* opengl)
    {
        if (get()->scalarField)
        {
            Eigen::Vector2i resolution = get()->scalarField->getGrid()->getResolution();
            static std::vector<float> data;
            data.resize(resolution.prod() * 1);
            for (int i = 0; i < resolution.prod(); ++i)
                data[i] = static_cast<float>(get()->scalarField->getArray()->getValue(i).x());

            glBindTexture(GL_TEXTURE_2D, mTexture);

            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                resolution.x(), resolution.y(),
                GL_RED,
                GL_FLOAT,
                data.data());

            glBindTexture(GL_TEXTURE_2D, 0);
        }
        mTransferFunction.update(scene);
    }

    bool ColormapTextureGl::createDevice(OpenGL* opengl)
    {
        if (get()->scalarField)
        {
            glGenTextures(1, &mTexture);
            glBindTexture(GL_TEXTURE_2D, mTexture);
            // set the texture wrapping/filtering options (on the currently bound texture object)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            // load and generate the texture
            Eigen::Vector2i resolution = get()->scalarField->getGrid()->getResolution();

            static std::vector<float> data;
            data.resize(resolution.prod() * 1);
            for (int i = 0; i < resolution.prod(); ++i)
                data[i] = static_cast<float>(get()->scalarField->getArray()->getValue(i).x());
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, resolution.x(), resolution.y(), 0, GL_RED, GL_FLOAT, data.data());
            // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, resolution.x(), resolution.y(), 0, GL_RGBA, GL_FLOAT, data.data());

            glBindTexture(GL_TEXTURE_2D, 0);
        }
        if (!mTransferFunction.createDevice(opengl))
            return false;

        return true;
    }

    void ColormapTextureGl::releaseDevice()
    {
        glDeleteTextures(1, &mTexture);
        mTransferFunction.releaseDevice();
    }

    bool ColormapTextureGl::bind(unsigned int shaderProgram, int textureBindingPoint, int keyBindingPoint, int valuesBindingPoint, int cbBindingPoint)
    {
        if (get()->scalarField)
        {
            glActiveTexture(GL_TEXTURE0 + textureBindingPoint);
            glBindTexture(GL_TEXTURE_2D, mTexture);
            glUniform1i(glGetUniformLocation(shaderProgram, "texColormap"), textureBindingPoint);
        }

        if (!mTransferFunction.bind(shaderProgram, keyBindingPoint, valuesBindingPoint, cbBindingPoint))
            return false;

        return true;
    }
    std::string ColormapTextureGl::generateCode() noexcept
    {
        return TransferFunction4dGl::generateCode();
    }
}
