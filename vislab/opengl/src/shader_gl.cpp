#include <iostream>
#include <vislab/opengl/opengl.hpp>
#include <vislab/opengl/shader_gl.hpp>

namespace vislab
{
    std::unordered_map<std::string, std::pair<unsigned int, unsigned int>> ShaderGl::gShaderCache;
    std::unordered_map<unsigned int, std::pair<unsigned int, unsigned int>> ShaderGl::gProgramCache;

    static unsigned int cantor(unsigned int a, unsigned int b)
    {
        return (a + b + 1) * (a + b) / 2 + b;
    }

    static unsigned int hash(unsigned int a, unsigned int b, unsigned int c)
    {
        return cantor(a, cantor(b, c));
    }

    ShaderGl::ShaderGl()
        : mVertexShader(0)
        , mGeometryShader(0)
        , mPixelShader(0)
        , mProgram(0)
    {
    }

    ShaderGl::~ShaderGl()
    {
        releaseDevice();
    }

    bool ShaderGl::createDevice()
    {
        std::string error;

        // compile vertex shader
        if (!sourceCode.vertexShader.empty())
        {
            auto it = gShaderCache.find(sourceCode.vertexShader);
            if (it != gShaderCache.end())
            {
                mVertexShader = it->second.first; // get the handle
                it->second.second++;              // increment reference counter
            }
            else
            {
                mVertexShader = compileShader(sourceCode.vertexShader.c_str(), GL_VERTEX_SHADER, &error);
                if (!error.empty())
                    std::cout << "ERROR: Vertex shader compilation failed.\n"
                              << error << std::endl
                              << sourceCode.vertexShader << std::endl;

                gShaderCache.insert(std::make_pair(sourceCode.vertexShader, std::make_pair(mVertexShader, 1)));
            }
        }

        // compile geometry shader
        if (!sourceCode.geometryShader.empty())
        {
            auto it = gShaderCache.find(sourceCode.geometryShader);
            if (it != gShaderCache.end())
            {
                mGeometryShader = it->second.first; // get the handle
                it->second.second++;                // increment reference counter
            }
            else
            {
                mGeometryShader = compileShader(sourceCode.geometryShader.c_str(), GL_GEOMETRY_SHADER, &error);
                if (!error.empty())
                    std::cout << "ERROR: Geometry shader compilation failed.\n"
                              << error << std::endl
                              << sourceCode.geometryShader << std::endl;

                gShaderCache.insert(std::make_pair(sourceCode.geometryShader, std::make_pair(mGeometryShader, 1)));
            }
        }

        // compile pixel shader
        if (!sourceCode.pixelShader.empty())
        {
            auto it = gShaderCache.find(sourceCode.pixelShader);
            if (it != gShaderCache.end())
            {
                mPixelShader = it->second.first; // get the handle
                it->second.second++;             // increment reference counter
            }
            else
            {
                mPixelShader = compileShader(sourceCode.pixelShader.c_str(), GL_FRAGMENT_SHADER, &error);
                if (!error.empty())
                    std::cout << "ERROR: Pixel shader compilation failed.\n"
                              << error << std::endl
                              << sourceCode.pixelShader << std::endl;

                gShaderCache.insert(std::make_pair(sourceCode.pixelShader, std::make_pair(mPixelShader, 1)));
            }
        }

        // link program
        auto it = gProgramCache.find(hash(mVertexShader, mGeometryShader, mPixelShader));
        if (it != gProgramCache.end())
        {
            mProgram = it->second.first;
            it->second.second++;
        }
        else
        {
            mProgram = linkProgram(&error);
            if (!error.empty())
                std::cout << "ERROR: Shader linkage failed.\n"
                          << error << std::endl;

            gProgramCache.insert(std::make_pair(hash(mVertexShader, mGeometryShader, mPixelShader), std::make_pair(mProgram, 1)));
        }
        return mProgram;
    }

    void ShaderGl::releaseDevice()
    {
        auto it = gProgramCache.find(hash(mVertexShader, mGeometryShader, mPixelShader));
        if (it != gProgramCache.end())
        {
            it->second.second--;
            if (it->second.second == 0)
                gProgramCache.erase(it);
        }

        // delete the shaders
        if (!sourceCode.vertexShader.empty())
        {
            auto it = gShaderCache.find(sourceCode.vertexShader);
            if (it != gShaderCache.end())
            {
                it->second.second--;
                if (it->second.second == 0)
                    gShaderCache.erase(it);
            }

            glDeleteShader(mVertexShader);
            mVertexShader = 0;
        }
        if (!sourceCode.geometryShader.empty())
        {
            auto it = gShaderCache.find(sourceCode.geometryShader);
            if (it != gShaderCache.end())
            {
                it->second.second--;
                if (it->second.second == 0)
                    gShaderCache.erase(it);
            }

            glDeleteShader(mGeometryShader);
            mGeometryShader = 0;
        }
        if (!sourceCode.pixelShader.empty())
        {
            auto it = gShaderCache.find(sourceCode.pixelShader);
            if (it != gShaderCache.end())
            {
                it->second.second--;
                if (it->second.second == 0)
                    gShaderCache.erase(it);
            }

            glDeleteShader(mPixelShader);
            mPixelShader = 0;
        }

        glDeleteProgram(mProgram);
        mProgram = 0;
    }

    void ShaderGl::bind()
    {
        glUseProgram(mProgram);
    }

    unsigned int ShaderGl::getProgram() const
    {
        return mProgram;
    }

    unsigned int ShaderGl::compileShader(const char* shaderSource, unsigned int shaderType, std::string* error)
    {
        unsigned int handle = glCreateShader(shaderType);
        glShaderSource(handle, 1, &shaderSource, NULL);
        glCompileShader(handle);
        int success;
        glGetShaderiv(handle, GL_COMPILE_STATUS, &success);
        if (!success && error)
        {
            int infologLength = 0;
            glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &infologLength);
            error->resize(infologLength);
            glGetShaderInfoLog(handle, infologLength, NULL, error->data());
        }
        return handle;
    }

    unsigned int ShaderGl::linkProgram(std::string* error)
    {
        unsigned int handle = glCreateProgram();
        if (mVertexShader)
            glAttachShader(handle, mVertexShader);
        if (mGeometryShader)
            glAttachShader(handle, mGeometryShader);
        if (mPixelShader)
            glAttachShader(handle, mPixelShader);
        glLinkProgram(handle);
        int success;
        glGetProgramiv(handle, GL_LINK_STATUS, &success);
        if (!success && error)
        {
            int infologLength = 0;
            glGetProgramiv(handle, GL_INFO_LOG_LENGTH, &infologLength);
            error->resize(infologLength);
            glGetProgramInfoLog(handle, infologLength, NULL, error->data());
        }
        return handle;
    }
}
