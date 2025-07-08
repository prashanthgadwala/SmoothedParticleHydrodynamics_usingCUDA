# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/cip/ce/yl34esew/Documents/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-src"
  "/home/cip/ce/yl34esew/Documents/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-build"
  "/home/cip/ce/yl34esew/Documents/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-subbuild/glew-populate-prefix"
  "/home/cip/ce/yl34esew/Documents/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-subbuild/glew-populate-prefix/tmp"
  "/home/cip/ce/yl34esew/Documents/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-subbuild/glew-populate-prefix/src/glew-populate-stamp"
  "/home/cip/ce/yl34esew/Documents/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-subbuild/glew-populate-prefix/src"
  "/home/cip/ce/yl34esew/Documents/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-subbuild/glew-populate-prefix/src/glew-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/cip/ce/yl34esew/Documents/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-subbuild/glew-populate-prefix/src/glew-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/cip/ce/yl34esew/Documents/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-subbuild/glew-populate-prefix/src/glew-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
