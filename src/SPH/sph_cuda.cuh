#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <Eigen/Dense>

// CUDA kernel declarations
extern "C" {
    // Density computation kernel
    void cuda_compute_density(
        const float* positions,     // particle positions (x,y,z interleaved)
        const float* boundary_positions, // boundary particle positions
        const float* masses,        // particle masses
        const float* boundary_masses, // boundary particle masses
        float* densities,           // output densities
        int num_particles,
        int num_boundary_particles,
        float support_radius,
        float rest_density
    );

    // Pressure computation kernel
    void cuda_compute_pressure(
        const float* densities,
        float* pressures,
        int num_particles,
        float stiffness,
        float rest_density,
        float exponent
    );

    // Force computation kernel
    void cuda_compute_forces(
        const float* positions,
        const float* velocities,
        const float* densities,
        const float* pressures,
        const float* masses,
        const float* boundary_positions,
        const float* boundary_masses,
        float* accelerations,
        int num_particles,
        int num_boundary_particles,
        float support_radius,
        float viscosity
    );

    // Particle integration kernel
    void cuda_integrate_particles(
        float* positions,
        float* velocities,
        const float* accelerations,
        const float* gravity,
        int num_particles,
        float dt
    );
}

// CUDA utility functions
namespace cuda_utils {
    // Check CUDA errors
    void checkCudaError(cudaError_t error, const char* file, int line);
    
    // Initialize CUDA context
    bool initCuda();
    
    // Cleanup CUDA resources
    void cleanupCuda();
}

// Macro for CUDA error checking
#define CUDA_CHECK(err) cuda_utils::checkCudaError(err, __FILE__, __LINE__)

// CUDA memory management helper class
class CudaMemoryManager {
public:
    CudaMemoryManager();
    ~CudaMemoryManager();
    
    // Allocate device memory
    void* allocateDevice(size_t size);
    
    // Free device memory
    void freeDevice(void* ptr);
    
    // Copy host to device
    void copyHostToDevice(void* device_ptr, const void* host_ptr, size_t size);
    
    // Copy device to host
    void copyDeviceToHost(void* host_ptr, const void* device_ptr, size_t size);
    
private:
    std::vector<void*> allocated_ptrs;
};
