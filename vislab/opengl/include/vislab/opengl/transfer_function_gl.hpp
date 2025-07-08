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
    class TransferFunction4dGl : public Concrete<TransferFunction4dGl>
    {
    public:
        /**
         * @brief Constructor.
         * @param transferFunction Transfer function to wrap.
         */
        TransferFunction4dGl(const TransferFunction4d& transferFunction);

        /**
         * @brief Destructor. Releases the opengl resources.
         */
        virtual ~TransferFunction4dGl();

        /**
         * @brief Updates the content of the bsdf.
         * @param scene Scene that holds the wrappers to the different resources.
         */
        void update(SceneGl* scene);

        /**
         * @brief Creates the device resources.
         * @param opengl Reference to the openGL handle.
         * @return True, if the creation succeeded.
         */
        [[nodiscard]] bool createDevice(OpenGL* opengl);

        /**
         * @brief Releases the device resources.
         */
        void releaseDevice();

        /**
         * @brief Binds the constant buffer for rendering.
         * @param shaderProgram Shader program handle to bind the parameters to.
         * @param keyBindingPoint Binding point to bind shader storage buffer with keys to.
         * @param valuesBindingPoint Binding point to bind shader storage buffer with values to.
         * @param cbBindingPoint Binding point to bind constant buffer to.
         * @return True, if binding was successful.
         */
        [[nodiscard]] bool bind(unsigned int shaderProgram, int keyBindingPoint, int valuesBindingPoint, int cbBindingPoint);

        /**
         * @brief Generates the shader code for working with this object.
         * @return String containing the generated shader code.
         */
        [[nodiscard]] static std::string generateCode() noexcept;

    private:
        /**
         * @brief Reference to the transfer function that is wrapped.
         */
        const TransferFunction4d& mTransferFunction;

        /**
         * @brief Uniform buffer parameters.
         */
        struct Params
        {
            /**
             * @brief Lower bound for mapping from [minValue, maxValue] -> [RGB]
             */
            float minValue;
            /**
             * @brief Upper bound for mapping from [minValue, maxValue] -> [RGB]
             */
            float maxValue;
            /**
             * @brief Number of values in the color map.
             */
            int numValues;
        };

        /**
         * @brief Wrapper that holds a uniform buffer.
         */
        vislab::ConstantBufferGl<Params> mParams;

        /**
         * @brief Keys of the transfer function.
         */
        StorageBufferGl<Eigen::Vector1f> mTransferFunctionKeys;

        /**
         * @brief Values of the transfer function.
         */
        StorageBufferGl<Eigen::Vector4f> mTransferFunctionValues;
    };
}
