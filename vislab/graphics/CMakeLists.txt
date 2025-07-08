# CMake Library Setup
# --------------------------------------------------
set(LIBRARY_NAME_SHORT graphics)
set(LIBRARY_NAME vislab_${LIBRARY_NAME_SHORT})

# Find source files
file(GLOB SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Create library
add_library(${LIBRARY_NAME} ${SOURCES})

# link
target_link_libraries(${LIBRARY_NAME} PUBLIC vislab_geometry vislab_field)

# include directories
# public:
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# private:
target_include_directories(${LIBRARY_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# alias target
add_library(vislab::${LIBRARY_NAME_SHORT} ALIAS ${LIBRARY_NAME})

# External Dependency Setup
# --------------------------------------------------

# Test Setup
# --------------------------------------------------
if(VISLAB_BUILD_TESTS)
    add_subdirectory(test)
endif()

# IDE Support
# --------------------------------------------------
if (VISLAB_IDE_SUPPORT)
    enable_ide_support_vislab(${LIBRARY_NAME} ${LIBRARY_NAME_SHORT} "" ${CMAKE_CURRENT_SOURCE_DIR})
endif()