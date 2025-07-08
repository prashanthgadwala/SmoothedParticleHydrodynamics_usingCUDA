#include <vislab/opengl/trimesh_geometry_gl.hpp>

#include <vislab/opengl/opengl.hpp>

#include <vislab/core/array.hpp>
#include <vislab/graphics/trimesh_geometry.hpp>

namespace vislab
{
    TrimeshGeometryGl::TrimeshGeometryGl(std::shared_ptr<const TrimeshGeometry> trimeshGeometry)
        : Concrete<TrimeshGeometryGl, GeometryGl>(trimeshGeometry, generateSourceCode(), false)
        , mVBO_positions(0)
        , mVBO_normals(0)
        , mIBO(0)
        , mVAO(0)
        , mNumIndices(0)
    {
    }

    TrimeshGeometryGl::~TrimeshGeometryGl()
    {
        this->releaseDevice();
    }

    void TrimeshGeometryGl::update(SceneGl* scene, OpenGL* opengl)
    {
        // get the mesh
        const TrimeshGeometry* mesh = get();

        if (!mesh->positions || !mesh->indices)
            throw std::logic_error("Vertex buffer and index buffer needed!");

        // update the positions
        glBindBuffer(GL_ARRAY_BUFFER, mVBO_positions);
        glBufferSubData(GL_ARRAY_BUFFER, 0, mesh->positions->getSizeInBytes(), mesh->positions->getData().data());

        // create VBO for normals
        if (mesh->normals)
        {
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_normals);
            glBufferSubData(GL_ARRAY_BUFFER, 0, mesh->normals->getSizeInBytes(), mesh->normals->getData().data());
        }

        // create VBO for normals
        if (mesh->data)
        {
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_data);
            glBufferSubData(GL_ARRAY_BUFFER, 0, mesh->data->getSizeInBytes(), mesh->data->getData().data());
        }

        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    bool TrimeshGeometryGl::createDevice(OpenGL* opengl)
    {
        // get the mesh
        const TrimeshGeometry* mesh = get();

        if (!mesh->positions || !mesh->indices)
            throw std::logic_error("Vertex buffer and index buffer needed!");

        mNumIndices = mesh->indices->getSize() * 3;

        // create VBO for positions
        glGenBuffers(1, &mVBO_positions);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO_positions);
        glBufferData(GL_ARRAY_BUFFER, mesh->positions->getSizeInBytes(), mesh->positions->getData().data(), GL_DYNAMIC_DRAW);

        // create VBO for normals
        if (mesh->normals)
        {
            if (mesh->positions->getSize() != mesh->normals->getSize())
                return false;

            glGenBuffers(1, &mVBO_normals);
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_normals);
            glBufferData(GL_ARRAY_BUFFER, mesh->normals->getSizeInBytes(), mesh->normals->getData().data(), GL_DYNAMIC_DRAW);
        }

        // create VBO for texCoords
        if (mesh->texCoords)
        {
            if (mesh->positions->getSize() != mesh->texCoords->getSize())
                return false;

            glGenBuffers(1, &mVBO_texCoords);
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_texCoords);
            glBufferData(GL_ARRAY_BUFFER, mesh->texCoords->getSizeInBytes(), mesh->texCoords->getData().data(), GL_DYNAMIC_DRAW);
        }

        // create VBO for data
        if (mesh->data)
        {
            if (mesh->positions->getSize() != mesh->data->getSize())
                return false;

            glGenBuffers(1, &mVBO_data);
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_data);
            glBufferData(GL_ARRAY_BUFFER, mesh->data->getSizeInBytes(), mesh->data->getData().data(), GL_DYNAMIC_DRAW);
        }

        // create IBO
        glGenBuffers(1, &mIBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->indices->getSizeInBytes(), mesh->indices->getData().data(), GL_DYNAMIC_DRAW);

        // create VAO
        glGenVertexArrays(1, &mVAO);
        glBindVertexArray(mVAO);
        //
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO_positions);
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);
        //
        if (mesh->normals)
        {
            glEnableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_normals);
            glVertexAttribPointer(1, 3, GL_FLOAT, false, 0, 0);
        }
        //
        if (mesh->texCoords)
        {
            glEnableVertexAttribArray(2);
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_texCoords);
            glVertexAttribPointer(2, 2, GL_FLOAT, false, 0, 0);
        }
        //
        if (mesh->data)
        {
            glEnableVertexAttribArray(3);
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_data);
            glVertexAttribPointer(3, 1, GL_FLOAT, false, 0, 0);
        }
        //
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
        glBindVertexArray(0);

        return true;
    }

    void TrimeshGeometryGl::releaseDevice()
    {
        glDeleteBuffers(1, &mVBO_positions);
        glDeleteBuffers(1, &mVBO_normals);
        glDeleteBuffers(1, &mVBO_texCoords);
        glDeleteBuffers(1, &mVBO_data);
        glDeleteBuffers(1, &mIBO);
        glDeleteVertexArrays(1, &mVAO);
    }

    void TrimeshGeometryGl::draw()
    {
        glBindVertexArray(mVAO);
        glDrawElements(GL_TRIANGLES, mNumIndices, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    bool TrimeshGeometryGl::bind(unsigned int shaderProgram, int bindingPoint)
    {
        return true;
    }

    const TrimeshGeometry* TrimeshGeometryGl::get() const
    {
        return static_cast<const TrimeshGeometry*>(GeometryGl::get());
    }

    ShaderSourceGl TrimeshGeometryGl::generateSourceCode()
    {
        ShaderSourceGl source;
        source.vertexShader =
            "layout (location = 0) in vec3 vs_in_Position;\n"
            "layout (location = 1) in vec3 vs_in_Normal;\n"
            "layout (location = 2) in vec2 vs_in_TexCoord;\n"
            "layout (location = 3) in float vs_in_Data;\n"
            "out vec3 vs_out_Position;\n"
            "out vec3 vs_out_Normal;\n"
            "out vec2 vs_out_TexCoord;\n"
            "out float vs_out_Data;\n"
            "vec4 transformLocalToWorld(vec4 localPosition);\n"
            "vec4 transformViewToClip(vec4 viewPosition);\n"
            "vec4 transformWorldToClip(vec4 worldPosition);\n"
            "vec3 transformNormalLocalToWorld(vec3 localNormal);\n"
            "void vs_geometry_default()\n"
            "{\n"
            "   vs_out_Position = transformLocalToWorld(vec4(vs_in_Position, 1)).xyz;\n"
            "   gl_Position = transformWorldToClip(vec4(vs_out_Position, 1));"
            "   vs_out_Normal = transformNormalLocalToWorld(vs_in_Normal);\n"
            "   vs_out_TexCoord = vs_in_TexCoord;\n"
            "   vs_out_Data = vs_in_Data;\n"
            "}\n";
        source.pixelShader =
            "in vec3 vs_out_Position;\n"
            "in vec3 vs_out_Normal;\n"
            "in vec2 vs_out_TexCoord;\n"
            "in float vs_out_Data;\n"
            "void ps_geometry_default(out vec3 fragPosition, out vec3 fragNormal, out vec2 fragTexCoord, out float fragData, out float fragDepth)\n"
            "{\n"
            "   fragPosition = vs_out_Position;\n"
            "   fragNormal = normalize(vs_out_Normal);\n"
            "   fragTexCoord = vs_out_TexCoord;\n"
            "   fragData = vs_out_Data;\n"
            "   fragDepth = 0;\n" // <-- not implemented yet
            "}\n";
        return source;
    }
}
