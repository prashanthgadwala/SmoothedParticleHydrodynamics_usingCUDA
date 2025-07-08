#pragma once

#include <string>

namespace vislab
{
    /**
     * @brief Holds shader code.
     */
    struct ShaderSourceGl
    {
        std::string vertexShader;   /* Vertex shader */
        std::string geometryShader; /* Geometry shader */
        std::string pixelShader;    /* Pixel shader */
    };
}
