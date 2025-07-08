#include <vislab/opengl/transfer_function_gl.hpp>

#include <vislab/opengl/colormap_texture_gl.hpp>
#include <vislab/opengl/opengl.hpp>
#include <vislab/opengl/scene_gl.hpp>

#include <vislab/graphics/colormap_texture.hpp>
#include <vislab/graphics/const_texture.hpp>
#include <vislab/graphics/diffuse_bsdf.hpp>
#include <vislab/graphics/texture.hpp>

namespace vislab
{
    TransferFunction4dGl::TransferFunction4dGl(const TransferFunction4d& transferFunction)
        : mTransferFunction(transferFunction)
    {
    }

    TransferFunction4dGl::~TransferFunction4dGl()
    {
        this->releaseDevice();
    }

    void TransferFunction4dGl::update(SceneGl* scene)
    {
        // assumes that the transfer function kept its size
        if (mTransferFunctionKeys.data->getSize() != mTransferFunction.values.size())
            return;
        if (mTransferFunctionValues.data->getSize() != mTransferFunction.values.size())
            return;

        mParams.data.minValue  = (float)mTransferFunction.minValue;
        mParams.data.maxValue  = (float)mTransferFunction.maxValue;
        mParams.data.numValues = (int)mTransferFunction.values.size();
        mParams.updateBuffer();

        int ivalue = 0;
        for (auto it = mTransferFunction.values.begin(); it != mTransferFunction.values.end(); ++it, ivalue++)
        {
            mTransferFunctionKeys.data->setValue(ivalue, Eigen::Vector1f(it->first));
            mTransferFunctionValues.data->setValue(ivalue, it->second.cast<float>());
        }
        mTransferFunctionKeys.updateBuffer();
        mTransferFunctionValues.updateBuffer();
    }

    bool TransferFunction4dGl::createDevice(OpenGL* opengl)
    {
        // allocate storage buffers for the transfer function
        mTransferFunctionKeys.data   = std::make_shared<Array1f>();
        mTransferFunctionValues.data = std::make_shared<Array4f>();
        mTransferFunctionKeys.data->setSize(mTransferFunction.values.size());
        mTransferFunctionValues.data->setSize(mTransferFunction.values.size());
        int ivalue = 0;
        for (auto it = mTransferFunction.values.begin(); it != mTransferFunction.values.end(); ++it, ivalue++)
        {
            mTransferFunctionKeys.data->setValue(ivalue, Eigen::Vector1f(it->first));
            mTransferFunctionValues.data->setValue(ivalue, it->second.cast<float>());
        }
        if (!mTransferFunctionKeys.createDevice())
            return false;
        if (!mTransferFunctionValues.createDevice())
            return false;

        mParams.data.minValue  = (float)mTransferFunction.minValue;
        mParams.data.maxValue  = (float)mTransferFunction.maxValue;
        mParams.data.numValues = (int)mTransferFunction.values.size();
        return mParams.createDevice();
    }

    void TransferFunction4dGl::releaseDevice()
    {
        mTransferFunctionKeys.releaseDevice();
        mTransferFunctionValues.releaseDevice();
        mParams.releaseDevice();
    }

    bool TransferFunction4dGl::bind(unsigned int shaderProgram, int keyBindingPoint, int valuesBindingPoint, int cbBindingPoint)
    {
        if (!mTransferFunctionKeys.bind(shaderProgram, "tfkey", keyBindingPoint))
            return false;
        if (!mTransferFunctionValues.bind(shaderProgram, "tfvalues", valuesBindingPoint))
            return false;
        return mParams.bind(shaderProgram, +"cbtf", cbBindingPoint);
    }

    std::string TransferFunction4dGl::generateCode() noexcept
    {
        return "layout(std140) uniform cbtf\n"
               "{\n"
               "   float tfMinValue;\n"
               "   float tfMaxValue;\n"
               "   int tfNumValues;\n"
               "};\n"
               "layout(std430) readonly buffer tfkey\n"
               "{\n"
               "   float tf_keys[];\n"
               "};\n"
               "layout(std430) readonly buffer tfvalues\n"
               "{\n"
               "   vec4 tf_values[];\n"
               "};\n"
               "vec4 transferFunction(float value)\n"
               "{\n"
               "   float t = min(max(0, (value - tfMinValue) / (tfMaxValue - tfMinValue)), 1);\n"
               "   int L = 0;\n"
               "   int R = tfNumValues - 1;\n"
               "   int istep = 0;\n"
               "   while (L < R && istep < tfNumValues) {\n"
               "      int m = (L + R) / 2;\n"
               "      float h = tf_keys[m];\n"
               "      if (h < t)\n"
               "         L = m + 1;\n"
               "      else\n"
               "         R = m;\n"
               "      istep++;\n"
               "   }\n"
               "   float h0 = tf_keys[max(0, L - 1)];\n"
               "   float h1 = tf_keys[L];\n"
               "   float rat = h1 == h0 ? 0 : ((t - h0) / (h1 - h0));\n"
               "   rat = min(max(0., rat), 1.);\n"
               "   return tf_values[max(0, L - 1)] * (1 - rat) + tf_values[L] * rat;\n"
               "}\n";
    }

}
