# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/nanoflann-src"
  "/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/nanoflann-build"
  "/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/nanoflann-subbuild/nanoflann-populate-prefix"
  "/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/nanoflann-subbuild/nanoflann-populate-prefix/tmp"
  "/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/nanoflann-subbuild/nanoflann-populate-prefix/src/nanoflann-populate-stamp"
  "/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/nanoflann-subbuild/nanoflann-populate-prefix/src"
  "/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/nanoflann-subbuild/nanoflann-populate-prefix/src/nanoflann-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/nanoflann-subbuild/nanoflann-populate-prefix/src/nanoflann-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/nanoflann-subbuild/nanoflann-populate-prefix/src/nanoflann-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
