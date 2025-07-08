#pragma once

#include <vislab/opengl/opengl.hpp>

#include <vislab/core/array.hpp>

namespace vislab
{
    /**
     * @brief Class that creates and manages an OpenGL shader storages buffer.
     * @tparam T Struct to create an array of TElement.
     */
    template <typename TElement>
    class StorageBufferGl
    {
    public:
        /**
         * @brief Constructor.
         */
        StorageBufferGl()
            : data()
            , mBuffer(-1)
        {
        }

        /**
         * @brief Copy-Constructor.
         * @param other Other buffer to copy the data from.
         */
        StorageBufferGl(const StorageBufferGl& other)
            : data(other.data)
            , mBuffer(-1)
        {
        }

        /**
         * @brief Destructor. Releases the OpenGL resource.
         */
        ~StorageBufferGl() { releaseDevice(); }

        /**
         * @brief Copy of the data on the CPU side.
         */
        std::shared_ptr<Array<TElement>> data;

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
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, mBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, data->getSizeInBytes(), data->getData().data(), GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
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
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, mBuffer);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, data->getSizeInBytes(), data->getData().data());
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
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
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, mBuffer);
            unsigned int blockIndex = glGetProgramResourceIndex(shaderProgram, GL_SHADER_STORAGE_BLOCK, blockName.c_str());
            if (blockIndex == GL_INVALID_INDEX)
                return false;
            glShaderStorageBlockBinding(shaderProgram, blockIndex, bindingPoint);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingPoint, mBuffer);
            return true;
        }

    private:
        /**
         * @brief Pointer to the OpenGL resource.
         */
        unsigned int mBuffer;
    };
}
