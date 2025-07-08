#include "init_core.hpp"

#include <vislab/core/reflect.hpp>

#include "vislab/core/algorithm.hpp"
#include "vislab/core/array.hpp"
#include "vislab/core/binary_input_archive.hpp"
#include "vislab/core/binary_output_archive.hpp"
#include "vislab/core/data.hpp"
#include "vislab/core/iarchive.hpp"
#include "vislab/core/iarray.hpp"
#include "vislab/core/iinput_archive.hpp"
#include "vislab/core/iinput_port.hpp"
#include "vislab/core/ioutput_archive.hpp"
#include "vislab/core/ioutput_port.hpp"
#include "vislab/core/iparameter.hpp"
#include "vislab/core/iserializable.hpp"
#include "vislab/core/numeric_parameter.hpp"
#include "vislab/core/object.hpp"
#include "vislab/core/option_parameter.hpp"
#include "vislab/core/parameter.hpp"
#include "vislab/core/path_parameter.hpp"

void init_core()
{
    using namespace vislab;

    // Core Bases

    reflect<Object>("Object");
    reflect<ISerializable>("ISerializable");
    reflect<Data, Object, ISerializable>("Data");
    reflect<IAlgorithm, Object, ISerializable>("IAlgorithm");

    // Interfaces

    reflect<IArchive, Object>("IArchive");
    reflect<IInputArchive, IArchive>("IInputArchive");
    reflect<IOutputArchive, IArchive>("IOutputArchive");
    reflect<IInputPort, Object>("IInputPort");
    reflect<IOutputPort, Object>("IOutputPort");
    reflect<ITransferFunction, Data>("ITransferFunction");
    reflect<IArray, Data>("IArray");
    reflect<IArray1, IArray>("IArray1");
    reflect<IArray2, IArray>("IArray2");
    reflect<IArray3, IArray>("IArray3");
    reflect<IArray4, IArray>("IArray4");
    reflect<IArray9, IArray>("IArray9");
    reflect<IArray16, IArray>("IArray16");
    reflect<IParameter, Data>("IParameter");

    // Constructibles

    reflect<Array1f, IArray1>("Array1f");
    reflect<Array2f, IArray2>("Array2f");
    reflect<Array3f, IArray3>("Array3f");
    reflect<Array4f, IArray4>("Array4f");
    reflect<Array2x2f, IArray4>("Array2x2f");
    reflect<Array3x3f, IArray9>("Array3x3f");
    reflect<Array4x4f, IArray16>("Array4x4f");

    reflect<Array1d, IArray1>("Array1d");
    reflect<Array2d, IArray2>("Array2d");
    reflect<Array3d, IArray3>("Array3d");
    reflect<Array4d, IArray4>("Array4d");
    reflect<Array2x2d, IArray4>("Array2x2d");
    reflect<Array3x3d, IArray9>("Array3x3d");
    reflect<Array4x4d, IArray16>("Array4x4d");

    reflect<Array1i_16, IArray1>("Array1i_16");
    reflect<Array2i_16, IArray2>("Array2i_16");
    reflect<Array3i_16, IArray3>("Array3i_16");
    reflect<Array4i_16, IArray4>("Array4i_16");
    reflect<Array2x2i_16, IArray4>("Array2x2i_16");
    reflect<Array3x3i_16, IArray9>("Array3x3i_16");
    reflect<Array4x4i_16, IArray16>("Array4x4i_16");

    reflect<Array1i, IArray1>("Array1i");
    reflect<Array2i, IArray2>("Array2i");
    reflect<Array3i, IArray3>("Array3i");
    reflect<Array4i, IArray4>("Array4i");
    reflect<Array2x2i, IArray4>("Array2x2i");
    reflect<Array3x3i, IArray9>("Array3x3i");
    reflect<Array4x4i, IArray16>("Array4x4i");

    reflect<Array1i_64, IArray1>("Array1i_64");
    reflect<Array2i_64, IArray2>("Array2i_64");
    reflect<Array3i_64, IArray3>("Array3i_64");
    reflect<Array4i_64, IArray4>("Array4i_64");
    reflect<Array2x2i_64, IArray4>("Array2x2i_64");
    reflect<Array3x3i_64, IArray9>("Array3x3i_64");
    reflect<Array4x4i_64, IArray16>("Array4x4i_64");

    reflect<Array1u_16, IArray1>("Array1u_16");
    reflect<Array2u_16, IArray2>("Array2u_16");
    reflect<Array3u_16, IArray3>("Array3u_16");
    reflect<Array4u_16, IArray4>("Array4u_16");
    reflect<Array2x2u_16, IArray4>("Array2x2u_16");
    reflect<Array3x3u_16, IArray9>("Array3x3u_16");
    reflect<Array4x4u_16, IArray16>("Array4x4u_16");

    reflect<Array1u, IArray1>("Array1u");
    reflect<Array2u, IArray2>("Array2u");
    reflect<Array3u, IArray3>("Array3u");
    reflect<Array4u, IArray4>("Array4u");
    reflect<Array2x2u, IArray4>("Array2x2u");
    reflect<Array3x3u, IArray9>("Array3x3u");
    reflect<Array4x4u, IArray16>("Array4x4u");

    reflect<Array1u_64, IArray1>("Array1ui_64");
    reflect<Array2u_64, IArray2>("Array2ui_64");
    reflect<Array3u_64, IArray3>("Array3ui_64");
    reflect<Array4u_64, IArray4>("Array4ui_64");
    reflect<Array2x2u_64, IArray4>("Array2x2u_64");
    reflect<Array3x3u_64, IArray9>("Array3x3u_64");
    reflect<Array4x4u_64, IArray16>("Array4x4u_64");

    reflect<BoolParameter, IParameter>("BoolParameter");
    reflect<StringParameter, IParameter>("StringParameter");

    reflect<FloatParameter, IParameter>("FloatParameter");
    reflect<DoubleParameter, IParameter>("DoubleParameter");
    reflect<Int32Parameter, IParameter>("Int32Parameter");
    reflect<Int64Parameter, IParameter>("Int64Parameter");
    reflect<Vec2iParameter, IParameter>("Vec2iParameter");
    reflect<Vec3iParameter, IParameter>("Vec3iParameter");
    reflect<Vec4iParameter, IParameter>("Vec4iParameter");
    reflect<Vec2fParameter, IParameter>("Vec2fParameter");
    reflect<Vec3fParameter, IParameter>("Vec3fParameter");
    reflect<Vec2dParameter, IParameter>("Vec2dParameter");
    reflect<Vec3dParameter, IParameter>("Vec3dParameter");
    reflect<Vec4dParameter, IParameter>("Vec4dParameter");

    // Vec4f and Color are the same type so just add it as alias
    reflect<Vec4fParameter, IParameter>("Vec4fParameter", { "ColorParameter" });

    reflect<PathParameter, IParameter>("PathParameter");
    reflect<OptionParameter, IParameter>("OptionParameter");

    reflect<BinaryInputArchive, IInputArchive>("BinaryInputArchive");
    reflect<BinaryOutputArchive, IOutputArchive>("BinaryOutputArchive");

    reflect<TransferFunction1f, ITransferFunction>("TransferFunction1f");
    reflect<TransferFunction1d, ITransferFunction>("TransferFunction1d");
    reflect<TransferFunction2f, ITransferFunction>("TransferFunction2f");
    reflect<TransferFunction2d, ITransferFunction>("TransferFunction2d");
    reflect<TransferFunction3f, ITransferFunction>("TransferFunction3f");
    reflect<TransferFunction3d, ITransferFunction>("TransferFunction3d");
    reflect<TransferFunction4f, ITransferFunction>("TransferFunction4f");
    reflect<TransferFunction4d, ITransferFunction>("TransferFunction4d");

    reflect<TransferFunction1fParameter, IParameter>("TransferFunction1fParameter");
    reflect<TransferFunction1dParameter, IParameter>("TransferFunction1dParameter");
    reflect<TransferFunction2fParameter, IParameter>("TransferFunction2fParameter");
    reflect<TransferFunction2dParameter, IParameter>("TransferFunction2dParameter");
    reflect<TransferFunction3fParameter, IParameter>("TransferFunction3fParameter");
    reflect<TransferFunction3dParameter, IParameter>("TransferFunction3dParameter");
    reflect<TransferFunction4fParameter, IParameter>("TransferFunction4fParameter");
    reflect<TransferFunction4dParameter, IParameter>("TransferFunction4dParameter");
}
