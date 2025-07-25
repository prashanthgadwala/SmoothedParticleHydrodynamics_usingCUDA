# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# compile CUDA with /usr/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_DEFINES = -DEIGEN_MATRIXBASE_PLUGIN=\"/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/vislab/core/eigenAddons/matrix_addons.hpp\" -DGLEW_NO_GLU -DGLEW_STATIC

CUDA_INCLUDES = --options-file CMakeFiles/sph_simulation.dir/includes_CUDA.rsp

CUDA_FLAGS =  --generate-code=arch=compute_60,code=[compute_60,sm_60] --generate-code=arch=compute_61,code=[compute_61,sm_61] --generate-code=arch=compute_70,code=[compute_70,sm_70] --generate-code=arch=compute_75,code=[compute_75,sm_75] --generate-code=arch=compute_80,code=[compute_80,sm_80] --generate-code=arch=compute_86,code=[compute_86,sm_86] -std=c++17

CXX_DEFINES = -DEIGEN_MATRIXBASE_PLUGIN=\"/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/vislab/core/eigenAddons/matrix_addons.hpp\" -DGLEW_NO_GLU -DGLEW_STATIC

CXX_INCLUDES = -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/src/SPH -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/src/SPH/generated -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/src/common/include -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/vislab/core/include -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/eigen3-src -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/meta-src/src -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/vislab/graphics/include -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/vislab/geometry/include -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/vislab/field/include -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/vislab/opengl/include -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-src/include -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glfw-src/include -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/imgui-src -I/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/nanoflann-src

CXX_FLAGS = -fopenmp -std=gnu++17

