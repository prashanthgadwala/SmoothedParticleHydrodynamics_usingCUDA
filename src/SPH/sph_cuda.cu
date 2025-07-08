#include "sph_cuda.cuh"
#include <stdio.h>
#include <math.h>
#include <algorithm>

// CUDA kernel parameters
#define BLOCK_SIZE 256
#define MAX_NEIGHBORS 64

// Cubic spline kernel implementation on GPU
__device__ float cubic_kernel(float r, float h) {
    const float pi = 3.14159265f;
    const float h3 = h * h * h;
    const float k = 8.0f / (pi * h3);
    const float q = r / h;
    
    if (q <= 1.0f) {
        if (q <= 0.5f) {
            const float q2 = q * q;
            const float q3 = q2 * q;
            return k * (6.0f * q3 - 6.0f * q2 + 1.0f);
        } else {
            const float temp = 2.0f - q;
            return k * 2.0f * temp * temp * temp;
        }
    }
    return 0.0f;
}

// Cubic spline kernel gradient implementation on GPU
__device__ void cubic_kernel_grad(float rx, float ry, float rz, float h, float* grad_x, float* grad_y, float* grad_z) {
    const float pi = 3.14159265f;
    const float h3 = h * h * h;
    const float l = 48.0f / (pi * h3);
    const float r = sqrtf(rx*rx + ry*ry + rz*rz);
    
    if (r > 1e-9f && r <= h) {
        const float q = r / h;
        float factor = 0.0f;
        
        if (q <= 0.5f) {
            factor = l * q * (3.0f * q - 2.0f) / h;
        } else if (q <= 1.0f) {
            const float temp = 2.0f - q;
            factor = -l * temp * temp / h;
        }
        
        const float inv_r = 1.0f / r;
        *grad_x = factor * rx * inv_r;
        *grad_y = factor * ry * inv_r;
        *grad_z = factor * rz * inv_r;
    } else {
        *grad_x = *grad_y = *grad_z = 0.0f;
    }
}

// CUDA kernel for density computation
__global__ void compute_density_kernel(
    const float* positions,
    const float* boundary_positions,
    const float* masses,
    const float* boundary_masses,
    float* densities,
    int num_particles,
    int num_boundary_particles,
    float support_radius,
    float rest_density
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float xi = positions[3*i];
    float yi = positions[3*i + 1];
    float zi = positions[3*i + 2];
    
    float density = 0.0f;
    
    // Compute density from fluid particles
    for (int j = 0; j < num_particles; j++) {
        float xj = positions[3*j];
        float yj = positions[3*j + 1];
        float zj = positions[3*j + 2];
        
        float dx = xi - xj;
        float dy = yi - yj;
        float dz = zi - zj;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        
        if (r <= support_radius) {
            float mass_j = masses[j];
            density += mass_j * cubic_kernel(r, support_radius);
        }
    }
    
    // Compute density from boundary particles
    for (int k = 0; k < num_boundary_particles; k++) {
        float xk = boundary_positions[3*k];
        float yk = boundary_positions[3*k + 1];
        float zk = boundary_positions[3*k + 2];
        
        float dx = xi - xk;
        float dy = yi - yk;
        float dz = zi - zk;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        
        if (r <= support_radius) {
            float mass_k = boundary_masses[k];
            density += mass_k * cubic_kernel(r, support_radius);
        }
    }
    
    densities[i] = fmaxf(density, rest_density);
}

// CUDA kernel for pressure computation
__global__ void compute_pressure_kernel(
    const float* densities,
    float* pressures,
    int num_particles,
    float stiffness,
    float rest_density,
    float exponent
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float rho = densities[i];
    float pressure = stiffness * (powf(rho / rest_density, exponent) - 1.0f);
    pressures[i] = fmaxf(pressure, 0.0f);
}

// CUDA kernel for force computation
__global__ void compute_forces_kernel(
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
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float xi = positions[3*i];
    float yi = positions[3*i + 1];
    float zi = positions[3*i + 2];
    
    float vxi = velocities[3*i];
    float vyi = velocities[3*i + 1];
    float vzi = velocities[3*i + 2];
    
    float pi = pressures[i];
    float rhoi = densities[i];
    
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    
    // Compute forces from fluid particles
    for (int j = 0; j < num_particles; j++) {
        if (i == j) continue;
        
        float xj = positions[3*j];
        float yj = positions[3*j + 1];
        float zj = positions[3*j + 2];
        
        float dx = xi - xj;
        float dy = yi - yj;
        float dz = zi - zj;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        
        if (r <= support_radius && r > 1e-9f) {
            float vxj = velocities[3*j];
            float vyj = velocities[3*j + 1];
            float vzj = velocities[3*j + 2];
            
            float pj = pressures[j];
            float rhoj = densities[j];
            float mass_j = masses[j];
            
            // Compute kernel gradient
            float grad_x, grad_y, grad_z;
            cubic_kernel_grad(dx, dy, dz, support_radius, &grad_x, &grad_y, &grad_z);
            
            // Pressure force
            float pressure_factor = -mass_j * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj));
            ax += pressure_factor * grad_x;
            ay += pressure_factor * grad_y;
            az += pressure_factor * grad_z;
            
            // Viscosity force
            float visc_factor = viscosity * mass_j / rhoj * cubic_kernel(r, support_radius) / r;
            ax += visc_factor * (vxj - vxi);
            ay += visc_factor * (vyj - vyi);
            az += visc_factor * (vzj - vzi);
        }
    }
    
    // Boundary forces (simplified Akinci boundary condition)
    for (int k = 0; k < num_boundary_particles; k++) {
        float xk = boundary_positions[3*k];
        float yk = boundary_positions[3*k + 1];
        float zk = boundary_positions[3*k + 2];
        
        float dx = xi - xk;
        float dy = yi - yk;
        float dz = zi - zk;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        
        if (r <= support_radius && r > 1e-9f) {
            float mass_k = boundary_masses[k];
            
            // Compute kernel gradient
            float grad_x, grad_y, grad_z;
            cubic_kernel_grad(dx, dy, dz, support_radius, &grad_x, &grad_y, &grad_z);
            
            // Boundary pressure force
            float boundary_factor = -mass_k * pi / (rhoi * rhoi);
            ax += boundary_factor * grad_x;
            ay += boundary_factor * grad_y;
            az += boundary_factor * grad_z;
        }
    }
    
    accelerations[3*i] = ax;
    accelerations[3*i + 1] = ay;
    accelerations[3*i + 2] = az;
}

// CUDA kernel for particle integration
__global__ void integrate_particles_kernel(
    float* positions,
    float* velocities,
    const float* accelerations,
    const float* gravity,
    int num_particles,
    float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    // Get current state
    float xi = positions[3*i];
    float yi = positions[3*i + 1];
    float zi = positions[3*i + 2];
    
    float vxi = velocities[3*i];
    float vyi = velocities[3*i + 1];
    float vzi = velocities[3*i + 2];
    
    float axi = accelerations[3*i] + gravity[0];
    float ayi = accelerations[3*i + 1] + gravity[1];
    float azi = accelerations[3*i + 2] + gravity[2];
    
    // Symplectic Euler integration
    float vxi_new = vxi + dt * axi;
    float vyi_new = vyi + dt * ayi;
    float vzi_new = vzi + dt * azi;
    
    float xi_new = xi + dt * vxi_new;
    float yi_new = yi + dt * vyi_new;
    float zi_new = zi + dt * vzi_new;
    
    // Update arrays
    velocities[3*i] = vxi_new;
    velocities[3*i + 1] = vyi_new;
    velocities[3*i + 2] = vzi_new;
    
    positions[3*i] = xi_new;
    positions[3*i + 1] = yi_new;
    positions[3*i + 2] = zi_new;
}

// Host function implementations
extern "C" {

void cuda_compute_density(
    const float* positions,
    const float* boundary_positions,
    const float* masses,
    const float* boundary_masses,
    float* densities,
    int num_particles,
    int num_boundary_particles,
    float support_radius,
    float rest_density
) {
    int blocks = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_density_kernel<<<blocks, BLOCK_SIZE>>>(
        positions, boundary_positions, masses, boundary_masses,
        densities, num_particles, num_boundary_particles,
        support_radius, rest_density
    );
    cudaDeviceSynchronize();
}

void cuda_compute_pressure(
    const float* densities,
    float* pressures,
    int num_particles,
    float stiffness,
    float rest_density,
    float exponent
) {
    int blocks = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_pressure_kernel<<<blocks, BLOCK_SIZE>>>(
        densities, pressures, num_particles,
        stiffness, rest_density, exponent
    );
    cudaDeviceSynchronize();
}

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
) {
    int blocks = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_forces_kernel<<<blocks, BLOCK_SIZE>>>(
        positions, velocities, densities, pressures, masses,
        boundary_positions, boundary_masses, accelerations,
        num_particles, num_boundary_particles, support_radius, viscosity
    );
    cudaDeviceSynchronize();
}

void cuda_integrate_particles(
    float* positions,
    float* velocities,
    const float* accelerations,
    const float* gravity,
    int num_particles,
    float dt
) {
    int blocks = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    integrate_particles_kernel<<<blocks, BLOCK_SIZE>>>(
        positions, velocities, accelerations, gravity, num_particles, dt
    );
    cudaDeviceSynchronize();
}

} // extern "C"

// CUDA utility functions implementation
namespace cuda_utils {

void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d - %s\n", file, line, cudaGetErrorString(error));
        exit(1);
    }
}

bool initCuda() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        printf("No CUDA devices found\n");
        return false;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using CUDA device: %s\n", prop.name);
    
    CUDA_CHECK(cudaSetDevice(0));
    return true;
}

void cleanupCuda() {
    cudaDeviceReset();
}

} // namespace cuda_utils

// CudaMemoryManager implementation
CudaMemoryManager::CudaMemoryManager() {}

CudaMemoryManager::~CudaMemoryManager() {
    for (void* ptr : allocated_ptrs) {
        cudaFree(ptr);
    }
}

void* CudaMemoryManager::allocateDevice(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    allocated_ptrs.push_back(ptr);
    return ptr;
}

void CudaMemoryManager::freeDevice(void* ptr) {
    auto it = std::find(allocated_ptrs.begin(), allocated_ptrs.end(), ptr);
    if (it != allocated_ptrs.end()) {
        CUDA_CHECK(cudaFree(ptr));
        allocated_ptrs.erase(it);
    }
}

void CudaMemoryManager::copyHostToDevice(void* device_ptr, const void* host_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice));
}

void CudaMemoryManager::copyDeviceToHost(void* host_ptr, const void* device_ptr, size_t size) {
    CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost));
}
