# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build

# Include any dependencies generated for this target.
include _deps/glew-build/CMakeFiles/libglew_static.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/glew-build/CMakeFiles/libglew_static.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/glew-build/CMakeFiles/libglew_static.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/glew-build/CMakeFiles/libglew_static.dir/flags.make

_deps/glew-build/CMakeFiles/libglew_static.dir/src/glew.c.o: _deps/glew-build/CMakeFiles/libglew_static.dir/flags.make
_deps/glew-build/CMakeFiles/libglew_static.dir/src/glew.c.o: _deps/glew-src/src/glew.c
_deps/glew-build/CMakeFiles/libglew_static.dir/src/glew.c.o: _deps/glew-build/CMakeFiles/libglew_static.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object _deps/glew-build/CMakeFiles/libglew_static.dir/src/glew.c.o"
	cd /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-build && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT _deps/glew-build/CMakeFiles/libglew_static.dir/src/glew.c.o -MF CMakeFiles/libglew_static.dir/src/glew.c.o.d -o CMakeFiles/libglew_static.dir/src/glew.c.o -c /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-src/src/glew.c

_deps/glew-build/CMakeFiles/libglew_static.dir/src/glew.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/libglew_static.dir/src/glew.c.i"
	cd /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-build && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-src/src/glew.c > CMakeFiles/libglew_static.dir/src/glew.c.i

_deps/glew-build/CMakeFiles/libglew_static.dir/src/glew.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/libglew_static.dir/src/glew.c.s"
	cd /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-build && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-src/src/glew.c -o CMakeFiles/libglew_static.dir/src/glew.c.s

# Object files for target libglew_static
libglew_static_OBJECTS = \
"CMakeFiles/libglew_static.dir/src/glew.c.o"

# External object files for target libglew_static
libglew_static_EXTERNAL_OBJECTS =

_deps/glew-build/lib/libglew.a: _deps/glew-build/CMakeFiles/libglew_static.dir/src/glew.c.o
_deps/glew-build/lib/libglew.a: _deps/glew-build/CMakeFiles/libglew_static.dir/build.make
_deps/glew-build/lib/libglew.a: _deps/glew-build/CMakeFiles/libglew_static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library lib/libglew.a"
	cd /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-build && $(CMAKE_COMMAND) -P CMakeFiles/libglew_static.dir/cmake_clean_target.cmake
	cd /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-build && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libglew_static.dir/link.txt --verbose=$(VERBOSE)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "create libGLEW symbolic link"
	cd /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-build/lib && /usr/bin/cmake -E create_symlink libglew.a libGLEW.a

# Rule to build all files generated by this target.
_deps/glew-build/CMakeFiles/libglew_static.dir/build: _deps/glew-build/lib/libglew.a
.PHONY : _deps/glew-build/CMakeFiles/libglew_static.dir/build

_deps/glew-build/CMakeFiles/libglew_static.dir/clean:
	cd /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-build && $(CMAKE_COMMAND) -P CMakeFiles/libglew_static.dir/cmake_clean.cmake
.PHONY : _deps/glew-build/CMakeFiles/libglew_static.dir/clean

_deps/glew-build/CMakeFiles/libglew_static.dir/depend:
	cd /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-src /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-build /home/cip/ce/yl34esew/Documents/SPH/SmoothedParticleHydrodynamics_usingCUDA/build/_deps/glew-build/CMakeFiles/libglew_static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/glew-build/CMakeFiles/libglew_static.dir/depend

