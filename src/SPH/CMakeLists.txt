set(EXECUTABLE_NAME sph_simulation)

# Add your project files
set(SRCFILES 
    main.cpp
    cuda_sph_simulation.cpp
    sph_kernels.cu
)

set(HFILES 
    cuda_sph_simulation.hpp
)

add_executable(${EXECUTABLE_NAME} ${SRCFILES} ${HFILES})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE physsim_common)
if(VISLAB_USE_CUDA)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE CUDA::cudart CUDA::curand)
endif()
target_include_directories(${EXECUTABLE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${EXECUTABLE_NAME} PROPERTIES FOLDER "src")

# Set CUDA properties
if(VISLAB_USE_CUDA)
    set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_ARCHITECTURES "60;61;70;75;80;86")
    set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    
    # Set language for .cu files
    set_source_files_properties(sph_kernels.cu PROPERTIES LANGUAGE CUDA)
endif()

include(../../cmake/generate_init.cmake)
generate_init_function(${EXECUTABLE_NAME})
