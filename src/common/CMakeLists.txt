set(LIBRARY_NAME physsim_common)

# Add your project files
file(GLOB SRCFILES src/*.cpp)
file(GLOB HFILES include/*.hpp)

include(cmake/imgui.cmake)
include(cmake/nanoflann.cmake)

add_library(${LIBRARY_NAME} ${SRCFILES} ${HFILES})
target_link_libraries(${LIBRARY_NAME} PUBLIC vislab_core vislab_graphics vislab_geometry vislab_field vislab_opengl imgui nanoflann)
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(${LIBRARY_NAME} PROPERTIES FOLDER "src")
