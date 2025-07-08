#include <vislab/opengl/rectangle_geometry_gl.hpp>

#include <vislab/opengl/opengl.hpp>

#include <vislab/graphics/rectangle_geometry.hpp>

namespace vislab
{
    RectangleGeometryGl::RectangleGeometryGl(std::shared_ptr<const RectangleGeometry> rectangleGeometry)
        : Concrete<RectangleGeometryGl, GeometryGl>(rectangleGeometry, generateSourceCode(), false)
        , mVAO(0)
    {
    }

    RectangleGeometryGl::~RectangleGeometryGl()
    {
        this->releaseDevice();
    }

    void RectangleGeometryGl::update(SceneGl* scene, OpenGL* opengl)
    {
    }

    bool RectangleGeometryGl::createDevice(OpenGL* opengl)
    {
        // create VBOs for position and texcoord
        GLuint rectangleVBOPos, rectangleVBOTexCoord;
        glGenBuffers(1, &rectangleVBOPos);
        glGenBuffers(1, &rectangleVBOTexCoord);

        // create VAO
        glGenVertexArrays(1, &mVAO);
        glBindVertexArray(mVAO);

        // position buffer
        glBindBuffer(GL_ARRAY_BUFFER, rectangleVBOPos);
        glEnableVertexAttribArray(0);
        float rectanglePos[] = { -1.f, -1.f, 0.f, 1.f, -1.f, 0.f, -1.f, 1.f, 0.f, 1.f, 1.f, 0.f };
        glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), &rectanglePos, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);

        // texcoord buffer
        glBindBuffer(GL_ARRAY_BUFFER, rectangleVBOTexCoord);
        glEnableVertexAttribArray(1);
        float rectangleTexCoord[] = { 0, 0, 1, 0, 0, 1, 1, 1 };
        glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), &rectangleTexCoord, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 2, GL_FLOAT, false, 0, 0);

        // unbind and return
        glBindVertexArray(0);
        return true;
    }

    void RectangleGeometryGl::releaseDevice()
    {
        glDeleteVertexArrays(1, &mVAO);
    }

    void RectangleGeometryGl::draw()
    {
        glBindVertexArray(mVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
    }

    bool RectangleGeometryGl::bind(unsigned int shaderProgram, int bindingPoint)
    {
        return true;
    }

    const RectangleGeometry* RectangleGeometryGl::get() const
    {
        return static_cast<const RectangleGeometry*>(GeometryGl::get());
    }

    ShaderSourceGl RectangleGeometryGl::generateSourceCode()
    {
        ShaderSourceGl source;
        source.vertexShader =
            "layout (location = 0) in vec3 Position;\n"
            "layout (location = 1) in vec2 TexCoord;\n"
            "out vec3 vs_out_Position;\n"
            "out vec2 vs_out_TexCoord;\n"
            "void vs_geometry_default()\n"
            "{\n"
            "   vs_out_Position = transformLocalToWorld(vec4(Position, 1)).xyz;\n"
            "   vs_out_TexCoord = TexCoord;\n"
            "   gl_Position = transformWorldToClip(vec4(vs_out_Position, 1));\n"
            "}\n";
        source.pixelShader =
            "in vec3 vs_out_Position;\n"
            "in vec2 vs_out_TexCoord;\n"
            "void ps_geometry_default(out vec3 fragPosition, out vec3 fragNormal, out vec2 fragTexCoord, out float fragData, out float fragDepth)\n"
            "{\n"
            "   fragPosition = vs_out_Position;\n"
            "   fragNormal = normalize((worldMatrix * vec4(0,0,1,0)).xyz);"
            "   fragTexCoord = vs_out_TexCoord;\n"
            "   fragData = 0;\n"
            "   fragDepth = 0;\n" // <-- not implemented yet
            "}\n";
        return source;
    }
}
