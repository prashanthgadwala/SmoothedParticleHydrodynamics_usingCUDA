#include "init_geometry.hpp"

#include <vislab/core/reflect.hpp>

#include "vislab/geometry/face_normals.hpp"
#include "vislab/geometry/iline.hpp"
#include "vislab/geometry/ilines.hpp"
#include "vislab/geometry/ipoints.hpp"
#include "vislab/geometry/isurface.hpp"
#include "vislab/geometry/isurfaces.hpp"
#include "vislab/geometry/line.hpp"
#include "vislab/geometry/lines.hpp"
#include "vislab/geometry/points.hpp"
#include "vislab/geometry/surface.hpp"
#include "vislab/geometry/surfaces.hpp"
#include "vislab/geometry/vertex_normals.hpp"

void init_geometry()
{
    using namespace vislab;

    // Interfaces

    reflect<ILine, Data>("ILine");
    reflect<ILines, Data>("ILines");
    reflect<IPoints, Data>("IPoints");
    reflect<ISurface, Data>("ISurface");
    reflect<ISurfaces, Data>("ISurfaces");

    reflect<ILine1, ILine>("ILine1");
    reflect<ILine2, ILine>("ILine2");
    reflect<ILine3, ILine>("ILine3");
    reflect<ILine4, ILine>("ILine4");

    reflect<ILines1, ILines>("ILines1");
    reflect<ILines2, ILines>("ILines2");
    reflect<ILines3, ILines>("ILines3");
    reflect<ILines4, ILines>("ILines4");

    reflect<IPoints1, IPoints>("IPoints1");
    reflect<IPoints2, IPoints>("IPoints2");
    reflect<IPoints3, IPoints>("IPoints3");
    reflect<IPoints4, IPoints>("IPoints4");

    reflect<ISurface1, ISurface>("ISurface1");
    reflect<ISurface2, ISurface>("ISurface2");
    reflect<ISurface3, ISurface>("ISurface3");
    reflect<ISurface4, ISurface>("ISurface4");

    reflect<ISurfaces1, ISurfaces>("ISurfaces1");
    reflect<ISurfaces2, ISurfaces>("ISurfaces2");
    reflect<ISurfaces3, ISurfaces>("ISurfaces3");
    reflect<ISurfaces4, ISurfaces>("ISurfaces4");

    // Constructibles

    reflect<Line1f, ILine1>("Line1f");
    reflect<Line2f, ILine2>("Line2f");
    reflect<Line3f, ILine3>("Line3f");
    reflect<Line4f, ILine4>("Line4f");

    reflect<Line1d, ILine1>("Line1d");
    reflect<Line2d, ILine2>("Line2d");
    reflect<Line3d, ILine3>("Line3d");
    reflect<Line4d, ILine4>("Line4d");

    reflect<Lines1f, ILines1>("Lines1f");
    reflect<Lines2f, ILines2>("Lines2f");
    reflect<Lines3f, ILines3>("Lines3f");
    reflect<Lines4f, ILines4>("Lines4f");

    reflect<Lines1d, ILines1>("Lines1d");
    reflect<Lines2d, ILines2>("Lines2d");
    reflect<Lines3d, ILines3>("Lines3d");
    reflect<Lines4d, ILines4>("Lines4d");

    reflect<Points1f, IPoints1>("Points1f");
    reflect<Points2f, IPoints2>("Points2f");
    reflect<Points3f, IPoints3>("Points3f");
    reflect<Points4f, IPoints4>("Points4f");

    reflect<Points1d, IPoints1>("Points1d");
    reflect<Points2d, IPoints2>("Points2d");
    reflect<Points3d, IPoints3>("Points3d");
    reflect<Points4d, IPoints4>("Points4d");

    reflect<Surface1f, ISurface1>("Surface1f");
    reflect<Surface2f, ISurface2>("Surface2f");
    reflect<Surface3f, ISurface3>("Surface3f");
    reflect<Surface4f, ISurface4>("Surface4f");

    reflect<Surface1d, ISurface1>("Surface1d");
    reflect<Surface2d, ISurface2>("Surface2d");
    reflect<Surface3d, ISurface3>("Surface3d");
    reflect<Surface4d, ISurface4>("Surface4d");

    reflect<Surfaces1f, ISurfaces1>("Surfaces1f");
    reflect<Surfaces2f, ISurfaces2>("Surfaces2f");
    reflect<Surfaces3f, ISurfaces3>("Surfaces3f");
    reflect<Surfaces4f, ISurfaces4>("Surfaces4f");

    reflect<Surfaces1d, ISurfaces1>("Surfaces1d");
    reflect<Surfaces2d, ISurfaces2>("Surfaces2d");
    reflect<Surfaces3d, ISurfaces3>("Surfaces3d");
    reflect<Surfaces4d, ISurfaces4>("Surfaces4d");

    reflect<FaceNormals3f, IAlgorithm>("FaceNormals3f")
        .member<&FaceNormals3f::inputSurfaces>("inputSurfaces", "Surfaces that the positions are read from.", "Surfaces")
        .member<&FaceNormals3f::outputSurfaces>("outputSurfaces", "Surface that the face normals are written to.", "Surfaces");

    reflect<VertexNormals3f, IAlgorithm>("VertexNormals3f")
        .member<&VertexNormals3f::inputSurfaces>("inputSurfaces", "Surfaces that the positions are read from.", "Surfaces")
        .member<&VertexNormals3f::outputSurfaces>("outputSurfaces", "Surface that the face normals are written to.", "Surfaces");
}
