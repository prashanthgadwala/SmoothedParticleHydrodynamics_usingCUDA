#pragma once

#include "bsdf_gl.hpp"
#include "storage_buffer_gl.hpp"

#include <vislab/opengl/constant_buffer_gl.hpp>

#include <Eigen/Eigen>

namespace vislab
{
    class DiffuseBSDF;
    class ColormapTextureGl;

    /**
     * @brief A BSDF for diffuse Lambertian shading.
     */
    class DiffuseBSDFGl : public Concrete<DiffuseBSDFGl, BSDFGl>
    {
    public:
        /**
         * @brief Constructor.
         * @param bsdf BSDF to wrap.
         */
        DiffuseBSDFGl(std::shared_ptr<const DiffuseBSDF> bsdf);

        /**
         * @brief Destructor. Releases the opengl resources.
         */
        virtual ~DiffuseBSDFGl();

        /**
         * @brief Updates the content of the bsdf.
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
         * @param textureBindingPoint Binding point to bind texture to.
         * @param keyBindingPoint Binding point to bind shader storage buffer with keys to.
         * @param valuesBindingPoint Binding point to bind shader storage buffer with values to.
         * @param cbBindingPointTf Binding point to bind constant buffer for the transfer function to.
         * @param cbBindingPointBsdf Binding point to bind constant buffer for the bsdf parameters to.
         * @return True, if binding was successful.
         */
        [[nodiscard]] bool bind(unsigned int shaderProgram, int textureBindingPoint, int keyBindingPoint, int valuesBindingPoint, int cbBindingPointTf, int cbBindingPointBsdf) override;

        /**
         * @brief Gets access to the underlying component that is wrapped.
         * @return Pointer to the underlying component.
         */
        [[nodiscard]] const DiffuseBSDF* get() const;

    private:
        /**
         * @brief Generates the shader code for working with this object.
         * @return String containing the generated shader code.
         */
        [[nodiscard]] static std::string generateShaderCode() noexcept;

        /**
         * @brief Enumeration of reflectance sources.
         */
        enum EReflectance : int32_t
        {
            Constant        = 0,
            FromData        = 1,
            FromScalarField = 2,
        };

        /**
         * @brief Uniform buffer parameters.
         */
        struct Params
        {
            /**
             * @brief Diffuse color.
             */
            Eigen::Vector4f diffuse;
            /**
             * @brief Flag that determines whether a diffuse texture is in use.
             */
            EReflectance reflectance;
        };

        /**
         * @brief Wrapper that holds a uniform buffer.
         */
        vislab::ConstantBufferGl<Params> mParams;

        /**
         * @brief Wrapper that holds a colormap texture.
         */
        std::shared_ptr<ColormapTextureGl> mColormapTexture;
    };
}
