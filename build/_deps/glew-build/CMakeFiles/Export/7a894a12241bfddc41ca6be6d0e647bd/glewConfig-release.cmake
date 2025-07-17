#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "libglew_static" for configuration "Release"
set_property(TARGET libglew_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libglew_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libglew.a"
  )

list(APPEND _cmake_import_check_targets libglew_static )
list(APPEND _cmake_import_check_files_for_libglew_static "${_IMPORT_PREFIX}/lib/libglew.a" )

# Import target "libglew_shared" for configuration "Release"
set_property(TARGET libglew_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libglew_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libglew-shared.so.2.2.0"
  IMPORTED_SONAME_RELEASE "libglew-shared.so.2.2.0"
  )

list(APPEND _cmake_import_check_targets libglew_shared )
list(APPEND _cmake_import_check_files_for_libglew_shared "${_IMPORT_PREFIX}/lib/libglew-shared.so.2.2.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
