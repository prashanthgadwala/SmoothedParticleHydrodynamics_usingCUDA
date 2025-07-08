#include "cuda_sph_simulation.hpp"
#include <vislab/core/array.hpp>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(err) checkCudaError(err, __FILE__, __LINE__)

namespace physsim {

CudaSPHSimulation::CudaSPHSimulation() 
    : d_positions(nullptr), d_velocities(nullptr), d_accelerations(nullptr)
    , d_densities(nullptr), d_pressures(nullptr), d_masses(nullptr)
    , d_boundary_positions(nullptr), d_boundary_masses(nullptr), d_gravity(nullptr)
    , max_particles(0), max_boundary_particles(0)
    , num_particles(0), num_boundary_particles(0)
    , support_radius(0.1f), rest_density(1000.0f), stiffness(1000.0f)
    , exponent(7.0f), viscosity(0.01f), gravity(0, -9.81f, 0)
    , domain_min_x(-1.0f), domain_max_x(1.0f)
    , domain_min_y(-1.0f), domain_max_y(1.0f)
    , domain_min_z(-1.0f), domain_max_z(1.0f)
    , boundary_damping(0.8f)
{
}

CudaSPHSimulation::~CudaSPHSimulation() {
    deallocateDeviceMemory();
}

bool CudaSPHSimulation::initialize(int max_particles, int max_boundary_particles) {
    if (!initCuda()) {
        std::cerr << "Failed to initialize CUDA" << std::endl;
        return false;
    }
    
    this->max_particles = max_particles;
    this->max_boundary_particles = max_boundary_particles;
    
    // Allocate host memory
    h_positions.resize(3 * max_particles);
    h_velocities.resize(3 * max_particles);
    h_accelerations.resize(3 * max_particles);
    h_densities.resize(max_particles);
    h_pressures.resize(max_particles);
    h_masses.resize(max_particles);
    
    h_boundary_positions.resize(3 * max_boundary_particles);
    h_boundary_masses.resize(max_boundary_particles);
    
    allocateDeviceMemory();
    
    // Upload gravity to device
    float gravity_data[3] = {gravity.x(), gravity.y(), gravity.z()};
    copyHostToDevice(d_gravity, gravity_data, 3 * sizeof(float));
    
    std::cout << "CUDA SPH simulation initialized with " 
              << max_particles << " max particles and " 
              << max_boundary_particles << " max boundary particles" << std::endl;
    
    return true;
}

void CudaSPHSimulation::setParameters(float support_radius, float rest_density, float stiffness, 
                                     float exponent, float viscosity, const Eigen::Vector3f& gravity) {
    this->support_radius = support_radius;
    this->rest_density = rest_density;
    this->stiffness = stiffness;
    this->exponent = exponent;
    this->viscosity = viscosity;
    this->gravity = gravity;
    
    // Update gravity on device
    if (d_gravity) {
        float gravity_data[3] = {gravity.x(), gravity.y(), gravity.z()};
        copyHostToDevice(d_gravity, gravity_data, 3 * sizeof(float));
    }
}

void CudaSPHSimulation::setBoundaryDomain(float min_x, float max_x, float min_y, float max_y, 
                                         float min_z, float max_z, float damping) {
    domain_min_x = min_x;
    domain_max_x = max_x;
    domain_min_y = min_y;
    domain_max_y = max_y;
    domain_min_z = min_z;
    domain_max_z = max_z;
    boundary_damping = damping;
}

void CudaSPHSimulation::uploadParticleData(
    const std::shared_ptr<vislab::Array3f>& positions,
    const std::shared_ptr<vislab::Array3f>& velocities,
    const std::shared_ptr<vislab::Array1f>& masses
) {
    num_particles = static_cast<int>(positions->getSize());
    
    if (num_particles > max_particles) {
        std::cerr << "Number of particles exceeds maximum allocated" << std::endl;
        return;
    }
    
    // Convert Eigen arrays to float arrays
    copyEigenToFloat(positions, h_positions);
    copyEigenToFloat(velocities, h_velocities);
    
    // Copy masses
    for (int i = 0; i < num_particles; ++i) {
        h_masses[i] = masses->getValue(i).x();
    }
    
    // Upload to device
    copyHostToDevice(d_positions, h_positions.data(), 3 * num_particles * sizeof(float));
    copyHostToDevice(d_velocities, h_velocities.data(), 3 * num_particles * sizeof(float));
    copyHostToDevice(d_masses, h_masses.data(), num_particles * sizeof(float));
}

void CudaSPHSimulation::uploadBoundaryData(
    const std::shared_ptr<vislab::Array3f>& boundary_positions,
    const std::shared_ptr<vislab::Array1f>& boundary_masses
) {
    num_boundary_particles = static_cast<int>(boundary_positions->getSize());
    
    if (num_boundary_particles > max_boundary_particles) {
        std::cerr << "Number of boundary particles exceeds maximum allocated" << std::endl;
        return;
    }
    
    // Convert Eigen arrays to float arrays
    copyEigenToFloat(boundary_positions, h_boundary_positions);
    
    // Copy masses
    for (int i = 0; i < num_boundary_particles; ++i) {
        h_boundary_masses[i] = boundary_masses->getValue(i).x();
    }
    
    // Upload to device
    copyHostToDevice(d_boundary_positions, h_boundary_positions.data(), 3 * num_boundary_particles * sizeof(float));
    copyHostToDevice(d_boundary_masses, h_boundary_masses.data(), num_boundary_particles * sizeof(float));
}

void CudaSPHSimulation::step(float dt) {
    if (num_particles == 0) return;
    
    // Step 1: Compute densities
    cuda_compute_density(
        d_positions, d_boundary_positions, d_masses, d_boundary_masses,
        d_densities, num_particles, num_boundary_particles,
        support_radius, rest_density
    );
    
    // Step 2: Compute pressures
    cuda_compute_pressure(
        d_densities, d_pressures, num_particles,
        stiffness, rest_density, exponent
    );
    
    // Step 3: Compute forces
    cuda_compute_forces(
        d_positions, d_velocities, d_densities, d_pressures, d_masses,
        d_boundary_positions, d_boundary_masses, d_accelerations,
        num_particles, num_boundary_particles, support_radius, viscosity
    );
    
    // Step 4: Integrate particles
    cuda_integrate_particles(
        d_positions, d_velocities, d_accelerations, d_gravity,
        num_particles, dt
    );
    
    // Step 5: Enforce boundary collisions
    cuda_enforce_boundary_collisions(
        d_positions, d_velocities, num_particles,
        domain_min_x, domain_max_x,
        domain_min_y, domain_max_y,
        domain_min_z, domain_max_z,
        boundary_damping
    );
}

void CudaSPHSimulation::downloadResults(
    std::shared_ptr<vislab::Array3f>& positions,
    std::shared_ptr<vislab::Array3f>& velocities,
    std::shared_ptr<vislab::Array1f>& densities,
    std::shared_ptr<vislab::Array1f>& pressures
) {
    if (num_particles == 0) return;
    
    // Download from device
    copyDeviceToHost(h_positions.data(), d_positions, 3 * num_particles * sizeof(float));
    copyDeviceToHost(h_velocities.data(), d_velocities, 3 * num_particles * sizeof(float));
    copyDeviceToHost(h_densities.data(), d_densities, num_particles * sizeof(float));
    copyDeviceToHost(h_pressures.data(), d_pressures, num_particles * sizeof(float));
    
    // Convert back to Eigen arrays
    copyFloatToEigen(h_positions, positions);
    copyFloatToEigen(h_velocities, velocities);
    
    // Copy scalar arrays
    for (int i = 0; i < num_particles; ++i) {
        densities->setValue(i, h_densities[i]);
        pressures->setValue(i, h_pressures[i]);
    }
}

// CUDA Memory Management Functions
void* CudaSPHSimulation::allocateDevice(size_t size) {
    void* ptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    allocated_ptrs.push_back(ptr);
    return ptr;
}

void CudaSPHSimulation::freeDevice(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
        auto it = std::find(allocated_ptrs.begin(), allocated_ptrs.end(), ptr);
        if (it != allocated_ptrs.end()) {
            allocated_ptrs.erase(it);
        }
    }
}

void CudaSPHSimulation::copyHostToDevice(void* device_ptr, const void* host_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice));
}

void CudaSPHSimulation::copyDeviceToHost(void* host_ptr, const void* device_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost));
}

void CudaSPHSimulation::allocateDeviceMemory() {
    // Allocate particle data
    d_positions = static_cast<float*>(allocateDevice(3 * max_particles * sizeof(float)));
    d_velocities = static_cast<float*>(allocateDevice(3 * max_particles * sizeof(float)));
    d_accelerations = static_cast<float*>(allocateDevice(3 * max_particles * sizeof(float)));
    d_densities = static_cast<float*>(allocateDevice(max_particles * sizeof(float)));
    d_pressures = static_cast<float*>(allocateDevice(max_particles * sizeof(float)));
    d_masses = static_cast<float*>(allocateDevice(max_particles * sizeof(float)));
    
    // Allocate boundary data
    d_boundary_positions = static_cast<float*>(allocateDevice(3 * max_boundary_particles * sizeof(float)));
    d_boundary_masses = static_cast<float*>(allocateDevice(max_boundary_particles * sizeof(float)));
    
    // Allocate gravity
    d_gravity = static_cast<float*>(allocateDevice(3 * sizeof(float)));
}

void CudaSPHSimulation::deallocateDeviceMemory() {
    // Free all allocated memory
    for (void* ptr : allocated_ptrs) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    allocated_ptrs.clear();
}

void CudaSPHSimulation::copyEigenToFloat(const std::shared_ptr<vislab::Array3f>& eigen_array, std::vector<float>& float_array) {
    for (int i = 0; i < static_cast<int>(eigen_array->getSize()); ++i) {
        Eigen::Vector3f vec = eigen_array->getValue(i);
        float_array[3*i] = vec.x();
        float_array[3*i + 1] = vec.y();
        float_array[3*i + 2] = vec.z();
    }
}

void CudaSPHSimulation::copyFloatToEigen(const std::vector<float>& float_array, std::shared_ptr<vislab::Array3f>& eigen_array) {
    for (int i = 0; i < static_cast<int>(eigen_array->getSize()); ++i) {
        Eigen::Vector3f vec(float_array[3*i], float_array[3*i + 1], float_array[3*i + 2]);
        eigen_array->setValue(i, vec);
    }
}

bool CudaSPHSimulation::initCuda() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA devices found: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Get device properties
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "Using CUDA device: " << deviceProp.name << std::endl;
    
    // Set device
    CUDA_CHECK(cudaSetDevice(0));
    
    return true;
}

void CudaSPHSimulation::checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " - " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

} // namespace physsim
