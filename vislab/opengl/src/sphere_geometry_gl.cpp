#include <vislab/opengl/sphere_geometry_gl.hpp>

#include <vislab/opengl/opengl.hpp>

#include <vislab/geometry/attributes.hpp>
#include <vislab/geometry/points.hpp>
#include <vislab/graphics/sphere_geometry.hpp>

namespace vislab
{
    SphereGeometryGl::SphereGeometryGl(std::shared_ptr<const SphereGeometry> sphereGeometry)
        : Concrete<SphereGeometryGl, GeometryGl>(sphereGeometry, generateSourceCode(), true)
        , mVAO(0)
        , mVBO_position(0)
        , mVBO_radius(0)
        , mVBO_data(0)
        , mNumVertices(0)
    {
    }

    SphereGeometryGl::~SphereGeometryGl()
    {
        this->releaseDevice();
    }

    void SphereGeometryGl::update(SceneGl* scene, OpenGL* opengl)
    {
        auto sphereGeometry      = get();
        mParams.data.radiusScale = sphereGeometry->radiusScale;
        mParams.updateBuffer();

        // update the positions
        glBindBuffer(GL_ARRAY_BUFFER, mVBO_position);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sphereGeometry->positions->getSizeInBytes(), sphereGeometry->positions->getData().data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // update the data array
        if (sphereGeometry->data)
        {
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_data);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sphereGeometry->data->getSizeInBytes(), sphereGeometry->data->getData().data());
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        // update the radius array
        if (sphereGeometry->radius)
        {
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_radius);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sphereGeometry->radius->getSizeInBytes(), sphereGeometry->radius->getData().data());
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
    }

    bool SphereGeometryGl::createDevice(OpenGL* opengl)
    {
        auto sphereGeometry = get();
        mNumVertices        = sphereGeometry->positions->getSize();

        // create VBOs for position
        glGenBuffers(1, &mVBO_position);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO_position);
        glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float) * mNumVertices, sphereGeometry->positions->getData().data(), GL_DYNAMIC_DRAW);

        // radius buffer
        if (sphereGeometry->radius)
        {
            if (sphereGeometry->positions->getSize() != sphereGeometry->radius->getSize())
                return false;

            glGenBuffers(1, &mVBO_radius);
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_radius);
            glBufferData(GL_ARRAY_BUFFER, 1 * sizeof(float) * mNumVertices, sphereGeometry->radius->getData().data(), GL_DYNAMIC_DRAW);
            mParams.data.hasRadiusBuffer = 1;
        }
        else
            mParams.data.hasRadiusBuffer = 0;

        // data buffer
        if (sphereGeometry->data)
        {
            if (sphereGeometry->positions->getSize() != sphereGeometry->data->getSize())
                return false;

            glGenBuffers(1, &mVBO_data);
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_data);
            glBufferData(GL_ARRAY_BUFFER, 1 * sizeof(float) * mNumVertices, sphereGeometry->data->getData().data(), GL_DYNAMIC_DRAW);
        }

        // create VAO
        glGenVertexArrays(1, &mVAO);
        glBindVertexArray(mVAO);
        //
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO_position);
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);
        //
        if (sphereGeometry->radius)
        {
            glEnableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_radius);
            glVertexAttribPointer(1, 1, GL_FLOAT, false, 0, 0);
        }
        //
        if (sphereGeometry->data)
        {
            glEnableVertexAttribArray(2);
            glBindBuffer(GL_ARRAY_BUFFER, mVBO_data);
            glVertexAttribPointer(2, 1, GL_FLOAT, false, 0, 0);
        }
        glBindVertexArray(0);

        return mParams.createDevice();
    }

    void SphereGeometryGl::releaseDevice()
    {
        glDeleteBuffers(1, &mVBO_position);
        glDeleteBuffers(1, &mVBO_radius);
        glDeleteBuffers(1, &mVBO_data);
        glDeleteVertexArrays(1, &mVAO);
        mParams.releaseDevice();
    }

    void SphereGeometryGl::draw()
    {
        glBindVertexArray(mVAO);
        glDrawArrays(GL_POINTS, 0, mNumVertices);
        glBindVertexArray(0);
    }

    bool SphereGeometryGl::bind(unsigned int shaderProgram, int bindingPoint)
    {
        return mParams.bind(shaderProgram, "cbgeometry", bindingPoint);
    }

    const SphereGeometry* SphereGeometryGl::get() const
    {
        return static_cast<const SphereGeometry*>(GeometryGl::get());
    }

    ShaderSourceGl SphereGeometryGl::generateSourceCode()
    {
        ShaderSourceGl source;
        source.vertexShader =
            "layout(std140) uniform cbgeometry\n"
            "{\n"
            "	float radiusScale;\n"
            "	int hasRadiusBuffer;\n"
            "};\n"
            "layout (location = 0) in vec3 vs_in_Position;\n"
            "layout (location = 1) in float vs_in_Radius;\n"
            "layout (location = 2) in float vs_in_Data;\n"
            "out float vs_out_Radius;\n"
            "out float vs_out_Data;\n"
            "void vs_geometry_default()\n"
            "{\n"
            "   gl_Position = transformLocalToWorld(vec4(vs_in_Position, 1.0));\n"
            "   vs_out_Radius = radiusScale;\n"
            "   if (hasRadiusBuffer == 1)\n"
            "      vs_out_Radius *= vs_in_Radius;\n"
            "   vs_out_Data = vs_in_Data;\n"
            "}\n";
        source.geometryShader =
            "layout(points) in;\n"
            "layout(triangle_strip, max_vertices = 4) out;\n"
            "in float vs_out_Radius[];\n"
            "in float vs_out_Data[];\n"
            "out vec4 gs_out_Position;\n"
            "out vec3 gs_out_Center;\n"
            "out float gs_out_Radius;\n"
            "out float gs_out_Data;\n"
            "void gs_geometry_default()\n"
            "{\n"
            "   gs_out_Center = gl_in[0].gl_Position.xyz;\n"
            "\n"
            "   float splatRadius = vs_out_Radius[0] * sqrt(2);\n"
            "   gs_out_Radius = vs_out_Radius[0];\n"
            "   gs_out_Data = vs_out_Data[0];\n"
            "   vec3 eye = normalize(eyePosition.xyz - gs_out_Center);\n"
            "   vec4 viewPos = transformWorldToView(vec4(gs_out_Center + eye * splatRadius, 1));\n"
            "   viewPos /= viewPos.w;\n"
            "\n"
            "   gl_Position = projMatrix * (viewPos + vec4(-1, -1, 0, 0) * splatRadius);\n"
            "   gs_out_Position = gl_Position;\n"
            "   EmitVertex();\n"
            "\n"
            "   gl_Position = projMatrix * (viewPos + vec4(1, -1, 0, 0) * splatRadius);\n"
            "   gs_out_Position = gl_Position;\n"
            "   EmitVertex();\n"
            "\n"
            "   gl_Position = projMatrix * (viewPos + vec4(-1, 1, 0, 0) * splatRadius);\n"
            "   gs_out_Position = gl_Position;\n"
            "   EmitVertex();\n"
            "\n"
            "   gl_Position = projMatrix * (viewPos + vec4(1, 1, 0, 0) * splatRadius);\n"
            "   gs_out_Position = gl_Position;\n"
            "   EmitVertex();\n"
            "   EndPrimitive();\n"
            "}\n";
        source.pixelShader =
            "layout(std140) uniform cbgeometry\n"
            "{\n"
            "	float radiusScale;\n"
            "	int hasRadiusBuffer;\n"
            "};\n"
            "in vec4 gs_out_Position;\n"
            "in vec3 gs_out_Center;\n"
            "in float gs_out_Radius;\n"
            "in float gs_out_Data;\n"
            "bool intersectRaySphere(vec3 o, vec3 l, vec3 c, float r, out vec2 d)\n"
            "{\n"
            "   vec3 oc = o - c;\n"
            "   float dloc = dot(l, oc);\n"
            "   float disc = pow(dloc, 2) - dot(oc, oc) + r * r;\n"
            "   if (disc < 0) return false;\n"
            "   float sqrtdisc = sqrt(disc);\n"
            "   d = vec2(-dloc - sqrtdisc, -dloc + sqrtdisc);\n"
            "   return true;\n"
            "}\n"
            "void ps_geometry_default(out vec3 fragPosition, out vec3 fragNormal, out vec2 fragTexCoord, out float fragData, out float fragDepth)\n"
            "{\n"
            "   vec4 npc = gs_out_Position;\n"
            "   npc /= npc.w;\n"
            "   vec4 npc0 = vec4(npc.xy, -1, 1);\n"
            "   vec4 npc1 = vec4(npc.xy, 1, 1);\n"
            "   vec4 world0 = transformClipToWorld(npc0);\n"
            "   vec4 world1 = transformClipToWorld(npc1);\n"
            "   vec3 origin = world0.xyz / world0.w;\n"
            "   vec3 dir = normalize(world1.xyz / world1.w - origin);\n"
            "   vec2 d;\n"
            "   if (!intersectRaySphere(origin, dir, gs_out_Center, gs_out_Radius, d))\n"
            "      discard;\n"
            "   float t = d.x < 0 ? d.y : d.x;\n"
            "   vec4 worldx = vec4(origin + dir * t, 1);\n"
            "   vec4 npcx = transformWorldToClip(worldx);\n"
            "   fragPosition = worldx.xyz;\n"
            "   fragNormal = normalize(worldx.xyz - gs_out_Center);\n"
            "   fragTexCoord = vec2(atan(fragNormal.z, fragNormal.x) / (2*3.1415926535) + 0.5, fragNormal.y * 0.5 + 0.5);\n"
            "   fragData = gs_out_Data;\n"
            "   fragDepth = npcx.z / npcx.w * 0.5 + 0.5;\n"
            "}\n";
        return source;
    }
}
