#include "init_field.hpp"

#include <vislab/core/reflect.hpp>

#include "vislab/field/ifield.hpp"
#include "vislab/field/regular_field.hpp"
#include "vislab/field/regular_grid.hpp"

void init_field()
{
    using namespace vislab;

    reflect<IField, Data>("IField");
    reflect<IGrid, Data>("IGrid");

    reflect<vislab::ISteadyScalarField2d, IField>("ISteadyScalarField2d");
    reflect<vislab::ISteadyScalarField3d, IField>("ISteadyScalarField3d");
    reflect<vislab::IUnsteadyScalarField2d, IField>("IUnsteadyScalarField2d");
    reflect<vislab::IUnsteadyScalarField3d, IField>("IUnsteadyScalarField3d");
    reflect<vislab::ISteadyVectorField2d, IField>("ISteadyVectorField2d");
    reflect<vislab::ISteadyVectorField3d, IField>("ISteadyVectorField3d");
    reflect<vislab::IUnsteadyVectorField2d, IField>("IUnsteadyVectorField2d");
    reflect<vislab::IUnsteadyVectorField3d, IField>("IUnsteadyVectorField3d");
    reflect<vislab::ISteadyTensorField2d, IField>("ISteadyTensorField2d");
    reflect<vislab::ISteadyTensorField3d, IField>("ISteadyTensorField3d");
    reflect<vislab::IUnsteadyTensorField2d, IField>("IUnsteadyTensorField2d");
    reflect<vislab::IUnsteadyTensorField3d, IField>("IUnsteadyTensorField3d");

    reflect<IGrid1, IGrid>("IGrid1");
    reflect<IGrid2, IGrid>("IGrid2");
    reflect<IGrid3, IGrid>("IGrid3");
    reflect<IGrid4, IGrid>("IGrid4");

    reflect<RegularSteadyScalarField2f, ISteadyScalarField2d>("RegularSteadyScalarField2f");
    reflect<RegularSteadyScalarField3f, ISteadyScalarField3d>("RegularSteadyScalarField3f");
    reflect<RegularUnsteadyScalarField2f, IUnsteadyScalarField2d>("RegularUnsteadyScalarField2f");
    reflect<RegularUnsteadyScalarField3f, IUnsteadyScalarField3d>("RegularUnsteadyScalarField3f");
    reflect<RegularSteadyVectorField2f, ISteadyVectorField2d>("RegularSteadyVectorField2f");
    reflect<RegularSteadyVectorField3f, ISteadyVectorField3d>("RegularSteadyVectorField3f");
    reflect<RegularUnsteadyVectorField2f, IUnsteadyVectorField2d>("RegularUnsteadyVectorField2f");
    reflect<RegularUnsteadyVectorField3f, IUnsteadyVectorField3d>("RegularUnsteadyVectorField3f");
    reflect<RegularSteadyTensorField2f, ISteadyTensorField2d>("RegularSteadyTensorField2f");
    reflect<RegularSteadyTensorField3f, ISteadyTensorField3d>("RegularSteadyTensorField3f");
    reflect<RegularUnsteadyTensorField2f, IUnsteadyTensorField2d>("RegularUnsteadyTensorField2f");
    reflect<RegularUnsteadyTensorField3f, IUnsteadyTensorField3d>("RegularUnsteadyTensorField3f");

    reflect<RegularSteadyScalarField2d, ISteadyScalarField2d>("RegularSteadyScalarField2d");
    reflect<RegularSteadyScalarField3d, ISteadyScalarField3d>("RegularSteadyScalarField3d");
    reflect<RegularUnsteadyScalarField2d, IUnsteadyScalarField2d>("RegularUnsteadyScalarField2d");
    reflect<RegularUnsteadyScalarField3d, IUnsteadyScalarField3d>("RegularUnsteadyScalarField3d");
    reflect<RegularSteadyVectorField2d, ISteadyVectorField2d>("RegularSteadyVectorField2d");
    reflect<RegularSteadyVectorField3d, ISteadyVectorField3d>("RegularSteadyVectorField3d");
    reflect<RegularUnsteadyVectorField2d, IUnsteadyVectorField2d>("RegularUnsteadyVectorField2d");
    reflect<RegularUnsteadyVectorField3d, IUnsteadyVectorField3d>("RegularUnsteadyVectorField3d");
    reflect<RegularSteadyTensorField2d, ISteadyTensorField2d>("RegularSteadyTensorField2d");
    reflect<RegularSteadyTensorField3d, ISteadyTensorField3d>("RegularSteadyTensorField3d");
    reflect<RegularUnsteadyTensorField2d, IUnsteadyTensorField2d>("RegularUnsteadyTensorField2d");
    reflect<RegularUnsteadyTensorField3d, IUnsteadyTensorField3d>("RegularUnsteadyTensorField3d");

    reflect<RegularGrid1d, IGrid1>("RegularGrid1d");
    reflect<RegularGrid2d, IGrid2>("RegularGrid2d");
    reflect<RegularGrid3d, IGrid3>("RegularGrid3d");
    reflect<RegularGrid4d, IGrid4>("RegularGrid4d");
}
