#pragma once

#include <vislab/opengl/opengl.hpp>

namespace vislab
{
    /**
     * @brief Class that creates and manages an OpenGL constant buffer.
     * @tparam T
     */
    template <typename T>
    class ConstantBufferGl
    {
    public:
        /**
         * @brief Constructor.
         */
        ConstantBufferGl()
            : data()
            , mBuffer(-1)
        {
        }

        /**
         * @brief Copy-Constructor.
         * @param other Other buffer to copy the data from.
         */
        ConstantBufferGl(const ConstantBufferGl& other)
            : data(other.data)
            , mBuffer(-1)
        {
        }

        /**
         * @brief Destructor. Releases the OpenGL resource.
         */
        ~ConstantBufferGl() { releaseDevice(); }

        /**
         * @brief Copy of the data on the CPU side.
         */
        T data;

        /**
         * @brief Gets the OpenGL resource handle.
         * @return OpenGL resource handle.
         */
        unsigned int getBuffer() const { return mBuffer; }

        /**
         * @brief Creates the OpenGL resource.
         * @return True, if creation succeeded.
         */
        [[nodiscard]] bool createDevice()
        {
            glGenBuffers(1, &mBuffer);
            glBindBuffer(GL_UNIFORM_BUFFER, mBuffer);
            glBufferData(GL_UNIFORM_BUFFER, sizeof(data), &data, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_UNIFORM_BUFFER, 0);
            return true;
        }

        /**
         * @brief Releases the OpenGL resource.
         */
        void releaseDevice()
        {
            glDeleteBuffers(1, &mBuffer);
            mBuffer = -1;
        }

        /**
         * @brief Maps the CPU content to the GPU.
         */
        void updateBuffer()
        {
            glBindBuffer(GL_UNIFORM_BUFFER, mBuffer);
            glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(T), &data);
            glBindBuffer(GL_UNIFORM_BUFFER, 0);
        }

        /**
         * @brief Binds the buffer for rendering.
         * @param shaderProgram Shader program handle to bind the parameters to.
         * @param blockName Name of the block to bind to.
         * @param bindingPoint Binding point to bind to.
         * @return True, if binding was successful.
         */
        [[nodiscard]] bool bind(unsigned int shaderProgram, const std::string& blockName, int bindingPoint)
        {
            glBindBuffer(GL_UNIFORM_BUFFER, mBuffer);
            unsigned int blockIndex = glGetUniformBlockIndex(shaderProgram, blockName.c_str());
            if (blockIndex == GL_INVALID_INDEX)
                return false;
            glUniformBlockBinding(shaderProgram, blockIndex, bindingPoint);
            glBindBufferBase(GL_UNIFORM_BUFFER, bindingPoint, mBuffer);
            return true;
        }

    private:
        /**
         * @brief Pointer to the OpenGL resource.
         */
        unsigned int mBuffer;
    };
}
