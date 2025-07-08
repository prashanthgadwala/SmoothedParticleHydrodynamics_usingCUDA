#pragma once

#include "sph_cuda.cuh"
#include <memory>
#include <vector>
#include <Eigen/Dense>

// Forward declarations
namespace vislab {
    template<typename T>
    class Array;
    using Array3f = Array<Eigen::Vector3f>;
    using Array1f = Array<float>;
}

namespace physsim {

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
    // Memory management
    std::unique_ptr<CudaMemoryManager> memory_manager;
    
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
    
    // Helper functions
    void allocateDeviceMemory();
    void deallocateDeviceMemory();
    void copyEigenToFloat(const std::shared_ptr<vislab::Array3f>& eigen_array, std::vector<float>& float_array);
    void copyFloatToEigen(const std::vector<float>& float_array, std::shared_ptr<vislab::Array3f>& eigen_array);
};

} // namespace physsim
