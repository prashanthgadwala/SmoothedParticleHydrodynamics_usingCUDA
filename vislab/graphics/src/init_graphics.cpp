#include "init_graphics.hpp"

#include <vislab/core/reflect.hpp>

#include "vislab/graphics/actor.hpp"
#include "vislab/graphics/area_light.hpp"
#include "vislab/graphics/bmp_reader.hpp"
#include "vislab/graphics/bmp_writer.hpp"
#include "vislab/graphics/bsdf.hpp"
#include "vislab/graphics/camera.hpp"
#include "vislab/graphics/colormap_texture.hpp"
#include "vislab/graphics/components.hpp"
#include "vislab/graphics/const_texture.hpp"
#include "vislab/graphics/dielectric_bsdf.hpp"
#include "vislab/graphics/diffuse_bsdf.hpp"
#include "vislab/graphics/free_interactor.hpp"
#include "vislab/graphics/geometry.hpp"
#include "vislab/graphics/iimage.hpp"
#include "vislab/graphics/iinteractor.hpp"
#include "vislab/graphics/image.hpp"
#include "vislab/graphics/light.hpp"
#include "vislab/graphics/medium.hpp"
#include "vislab/graphics/orthographic_camera.hpp"
#include "vislab/graphics/perspective_camera.hpp"
#include "vislab/graphics/point_light.hpp"
#include "vislab/graphics/projective_camera.hpp"
#include "vislab/graphics/rectangle_geometry.hpp"
#include "vislab/graphics/renderer.hpp"
#include "vislab/graphics/resource.hpp"
#include "vislab/graphics/sphere_geometry.hpp"
#include "vislab/graphics/texture.hpp"
#include "vislab/graphics/trackball_interactor.hpp"
#include "vislab/graphics/transform.hpp"
#include "vislab/graphics/trimesh_geometry.hpp"
#include "vislab/graphics/zoompan_interactor.hpp"

void init_graphics()
{
    using namespace vislab;

    reflect<Resource>("Resource");

    // Interfaces

    reflect<IImage, Data>("IImage");

    reflect<IImage1, IImage>("IImage1");
    reflect<IImage2, IImage>("IImage2");
    reflect<IImage3, IImage>("IImage3");
    reflect<IImage4, IImage>("IImage4");

    reflect<IInteractor, Data>("IInteractor");
    reflect<Component, Object, ISerializable, Resource>("Component");
    reflect<BSDF, Component>("BSDF");
    reflect<Geometry, Component>("Geometry");
    reflect<Camera, Component>("Camera");
    reflect<ProjectiveCamera, Camera>("ProjectiveCamera");
    reflect<Light, Component>("Light");
    reflect<Medium, Component>("Medium");
    reflect<Texture, Data>("Texture");

    // Constructibles

    reflect<Image1f, IImage1>("Image1f");
    reflect<Image2f, IImage2>("Image2f");
    reflect<Image3f, IImage3>("Image3f");
    reflect<Image4f, IImage4>("Image4f");

    reflect<Image1d, IImage1>("Image1d");
    reflect<Image2d, IImage2>("Image2d");
    reflect<Image3d, IImage3>("Image3d");
    reflect<Image4d, IImage4>("Image4d");

    reflect<Tag, Object>("Tag");
    reflect<Tags, Object, ISerializable>("Tags");
    reflect<Components, Object, ISerializable>("Components");
    reflect<Actor, Data, Resource>("Actor");
    reflect<DiffuseBSDF, BSDF>("DiffuseBSDF");
    reflect<DielectricBSDF, BSDF>("DielectricBSDF");
    reflect<RectangleGeometry, Geometry>("RectangleGeometry");
    reflect<SphereGeometry, Geometry>("SphereGeometry");
    reflect<TrimeshGeometry, Geometry>("TrimeshGeometry");
    reflect<Transform, Component>("Transform");
    reflect<Renderer, Object>("Renderer");
    reflect<ConstTexture, Texture>("ConstTexture");
    reflect<ColormapTexture, Texture>("ColormapTexture");
    reflect<PerspectiveCamera, ProjectiveCamera>("PerspectiveCamera");
    reflect<OrthographicCamera, ProjectiveCamera>("OrthographicCamera");
    reflect<PointLight, Light>("PointLight");
    reflect<AreaLight, Light>("AreaLight");
    reflect<TrackballInteractor, IInteractor>("TrackballInteractor");
    reflect<FreeInteractor, IInteractor>("FreeInteractor");
    reflect<ZoomPanInteractor, IInteractor>("ZoomPanInteractor");

    reflect<BmpReader, IAlgorithm>("BmpReader")
        .member<&BmpReader::outputImage>("outputImage", "Image data that was read from file.", "Image")
        .member<&BmpReader::paramPath>("paramPath", "Path to the file to read.", "Path");

    reflect<BmpWriter, IAlgorithm>("BmpWriter")
        .member<&BmpWriter::inputImage>("inputImage", "Image data to write to file.", "Image")
        .member<&BmpWriter::paramPath>("paramPath", "Path to the file to write.", "Path");
}
