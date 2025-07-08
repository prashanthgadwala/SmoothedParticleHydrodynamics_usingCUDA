#pragma once

#include "resource_gl.hpp"

#include "constant_buffer_gl.hpp"
#include "transfer_function_gl.hpp"

#include <vislab/graphics/colormap_texture.hpp>

namespace vislab
{
    /**
     * @brief Wrapper for a transformation object.
     */
    class ColormapTextureGl : public Concrete<ColormapTextureGl, ResourceGl<ColormapTexture>>
    {
    public:
        /**
         * @brief Constructor.
         * @param colormapTexture Colormap texture to be wrapped.
         */
        ColormapTextureGl(std::shared_ptr<const ColormapTexture> colormapTexture);

        /**
         * @brief Destructor. Releases the opengl resources.
         */
        virtual ~ColormapTextureGl();

        /**
         * @brief Updates this resource.
         * @param scene Scene that holds the wrappers to the different resources.
         * @param opengl Reference to the openGL handle.
         */
        void update(SceneGl* scene, OpenGL* opengl) override;

        /**
         * @brief Creates the device resources.
         * @param opengl Reference to the openGL handle.
         * @return True, if the creation succeeded.
         */
        [[nodiscard]] bool createDevice(OpenGL* opengl) override;

        /**
         * @brief Releases the device resources.
         */
        void releaseDevice() override;

        /**
         * @brief Binds the constant buffer for rendering.
         * @param shaderProgram Shader program handle to bind the parameters to.
         * @param textureBindingPoint Binding point for the texture.
         * @param keyBindingPoint Binding point to bind shader storage buffer with keys to.
         * @param valuesBindingPoint Binding point to bind shader storage buffer with values to.
         * @param cbBindingPoint Binding point to bind constant buffer to.
         * @return True, if binding was successful.
         */
        [[nodiscard]] bool bind(unsigned int shaderProgram, int textureBindingPoint, int keyBindingPoint, int valuesBindingPoint, int cbBindingPoint);

        /**
         * @brief Generates the shader code for working with this object.
         * @return String containing the generated shader code.
         */
        static std::string generateCode() noexcept;

    private:
        /**
         * @brief Handle for the texture that stores the field.
         */
        unsigned int mTexture;

        /**
         * @brief Wrapper for the transfer function.
         */
        TransferFunction4dGl mTransferFunction;
    };
}
