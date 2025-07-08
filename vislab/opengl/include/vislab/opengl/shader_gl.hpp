#pragma once

#include "shader_source_gl.hpp"

#include <unordered_map>

namespace vislab
{
    /**
     * @brief Represents an OpenGL shader.
     */
    class ShaderGl
    {
    public:
        /**
         * @brief Constructor.
         */
        ShaderGl();

        /**
         * @brief Destructor.
         */
        ~ShaderGl();

        /**
         * @brief Creates the resources that are independent of the viewport resolution.
         * @return
         */
        [[nodiscard]] bool createDevice();

        /**
         * @brief Releases the resources that are independent of the viewport resolution.
         */
        void releaseDevice();

        /**
         * @brief Bind the shader.
         */
        void bind();

        /**
         * @brief GLSL shader source code.
         */
        ShaderSourceGl sourceCode;

        /**
         * @brief Gets the OpenGL program handle.
         * @return Program handle.
         */
        [[nodiscard]] unsigned int getProgram() const;

    private:
        /**
         * @brief Cache of shaders, which maps: [shader code] -> [shader handle, number of alive instances]
         */
        static std::unordered_map<std::string, std::pair<unsigned int, unsigned int>> gShaderCache;

        /**
         * @brief Cache of programs, which maps: [vs handle, gs handle, ps handle] -> [program handle, number of alive instances]
         */
        static std::unordered_map<unsigned int, std::pair<unsigned int, unsigned int>> gProgramCache;

        /**
         * @brief Compiles a shader.
         * @param shaderSource Shader source to compile.
         * @param shaderType Shader type.
         * @param error Optional error message.
         * @return Shader handle.
         */
        [[nodiscard]] static unsigned int compileShader(const char* shaderSource, unsigned int shaderType, std::string* error = NULL);

        /**
         * @brief Link the shaders into a program.
         * @param error Optional error message.
         * @return Program handle.
         */
        [[nodiscard]] unsigned int linkProgram(std::string* error = NULL);

        /**
         * @brief Vertex shader handle.
         */
        unsigned int mVertexShader;

        /**
         * @brief Geometry shader handle.
         */
        unsigned int mGeometryShader;

        /**
         * @brief Fragement/pixel shader handle.
         */
        unsigned int mPixelShader;

        /**
         * @brief Program handle.
         */
        unsigned int mProgram;
    };
}
