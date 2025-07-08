#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <vislab/core/array_fwd.hpp>
#include <cuda_runtime.h>

namespace physsim {

// Forward declarations of CUDA kernels
extern "C" {
    void cuda_compute_density(
        const float* positions, const float* boundary_positions,
        const float* masses, const float* boundary_masses,
        float* densities, int num_particles, int num_boundary_particles,
        float support_radius, float rest_density
    );
    
    void cuda_compute_pressure(
        const float* densities, float* pressures, int num_particles,
        float stiffness, float rest_density, float exponent
    );
    
    void cuda_compute_forces(
        const float* positions, const float* velocities,
        const float* densities, const float* pressures, const float* masses,
        const float* boundary_positions, const float* boundary_masses,
        float* accelerations, int num_particles, int num_boundary_particles,
        float support_radius, float viscosity
    );
    
    void cuda_integrate_particles(
        float* positions, float* velocities, const float* accelerations,
        const float* gravity, int num_particles, float dt
    );
    
    void cuda_enforce_boundary_collisions(
        float* positions, float* velocities, int num_particles,
        float domain_min_x, float domain_max_x,
        float domain_min_y, float domain_max_y,
        float domain_min_z, float domain_max_z,
        float damping_factor
    );
}

/**
 * @brief CUDA-accelerated SPH simulation class
 */
class CudaSPHSimulation {
public:
    CudaSPHSimulation();
    ~CudaSPHSimulation();
    
    // Initialize CUDA simulation
    bool initialize(int max_particles, int max_boundary_particles);
    
    // Update simulation parameters
    void setParameters(float support_radius, float rest_density, float stiffness, 
                      float exponent, float viscosity, const Eigen::Vector3f& gravity);
    
    // Set boundary domain
    void setBoundaryDomain(float min_x, float max_x, float min_y, float max_y, 
                          float min_z, float max_z, float damping = 0.8f);
    
    // Upload particle data to GPU
    void uploadParticleData(
        const std::shared_ptr<vislab::Array3f>& positions,
        const std::shared_ptr<vislab::Array3f>& velocities,
        const std::shared_ptr<vislab::Array1f>& masses
    );
    
    // Upload boundary data to GPU
    void uploadBoundaryData(
        const std::shared_ptr<vislab::Array3f>& boundary_positions,
        const std::shared_ptr<vislab::Array1f>& boundary_masses
    );
    
    // Run one simulation step on GPU
    void step(float dt);
    
    // Download results from GPU
    void downloadResults(
        std::shared_ptr<vislab::Array3f>& positions,
        std::shared_ptr<vislab::Array3f>& velocities,
        std::shared_ptr<vislab::Array1f>& densities,
        std::shared_ptr<vislab::Array1f>& pressures
    );
    
    // Get current number of particles
    int getNumParticles() const { return num_particles; }
    int getNumBoundaryParticles() const { return num_boundary_particles; }

private:
    // CUDA memory management
    void* allocateDevice(size_t size);
    void freeDevice(void* ptr);
    void copyHostToDevice(void* device_ptr, const void* host_ptr, size_t size);
    void copyDeviceToHost(void* host_ptr, const void* device_ptr, size_t size);
    void allocateDeviceMemory();
    void deallocateDeviceMemory();
    
    // Utility functions
    void copyEigenToFloat(const std::shared_ptr<vislab::Array3f>& eigen_array, std::vector<float>& float_array);
    void copyFloatToEigen(const std::vector<float>& float_array, std::shared_ptr<vislab::Array3f>& eigen_array);
    bool initCuda();
    void checkCudaError(cudaError_t error, const char* file, int line);

    // Memory management
    std::vector<void*> allocated_ptrs;
    
    // Device memory pointers
    float* d_positions;
    float* d_velocities;
    float* d_accelerations;
    float* d_densities;
    float* d_pressures;
    float* d_masses;
    
    float* d_boundary_positions;
    float* d_boundary_masses;
    
    float* d_gravity;
    
    // Host memory buffers
    std::vector<float> h_positions;
    std::vector<float> h_velocities;
    std::vector<float> h_accelerations;
    std::vector<float> h_densities;
    std::vector<float> h_pressures;
    std::vector<float> h_masses;
    
    std::vector<float> h_boundary_positions;
    std::vector<float> h_boundary_masses;
    
    // Simulation parameters
    int max_particles;
    int max_boundary_particles;
    int num_particles;
    int num_boundary_particles;
    
    float support_radius;
    float rest_density;
    float stiffness;
    float exponent;
    float viscosity;
    Eigen::Vector3f gravity;
    
    // Boundary domain parameters
    float domain_min_x, domain_max_x;
    float domain_min_y, domain_max_y;
    float domain_min_z, domain_max_z;
    float boundary_damping;
};

} // namespace physsim
