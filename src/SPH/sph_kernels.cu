#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

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
    const float r = sqrtf(rx*rx + ry*ry + rz*rz);
    
    if (r > 1e-9f && r <= h) {
        const float q = r / h;
        float dW_dq = 0.0f;
        
        if (q <= 0.5f) {
            dW_dq = (8.0f / (pi * h3)) * (18.0f * q * q - 12.0f * q);
        } else if (q <= 1.0f) {
            const float temp = 2.0f - q;
            dW_dq = -(8.0f / (pi * h3)) * 6.0f * temp * temp;
        }
        
        const float factor = dW_dq / (h * r);
        *grad_x = factor * rx;
        *grad_y = factor * ry;
        *grad_z = factor * rz;
    } else {
        *grad_x = *grad_y = *grad_z = 0.0f;
    }
}

// Density computation kernel
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
    
    // Density contribution from fluid particles
    for (int j = 0; j < num_particles; j++) {
        float xj = positions[3*j];
        float yj = positions[3*j + 1];
        float zj = positions[3*j + 2];
        
        float dx = xi - xj;
        float dy = yi - yj;
        float dz = zi - zj;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        
        if (r <= support_radius) {
            float mass_j = (j < num_particles) ? masses[j] : 0.0f;
            density += mass_j * cubic_kernel(r, support_radius);
        }
    }
    
    // Density contribution from boundary particles
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

// Pressure computation kernel
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
    
    float density = densities[i];
    float pressure = stiffness * (powf(density / rest_density, exponent) - 1.0f);
    pressures[i] = fmaxf(pressure, 0.0f);
}

// Force computation kernel
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
    
    // Force computation with fluid particles
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
            float massj = masses[j];
            
            // Pressure gradient
            float grad_x, grad_y, grad_z;
            cubic_kernel_grad(dx, dy, dz, support_radius, &grad_x, &grad_y, &grad_z);
            
            float pressure_factor = massj * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj));
            ax -= pressure_factor * grad_x;
            ay -= pressure_factor * grad_y;
            az -= pressure_factor * grad_z;
            
            // Viscosity
            float viscosity_factor = viscosity * massj / rhoj * cubic_kernel(r, support_radius);
            ax += viscosity_factor * (vxj - vxi);
            ay += viscosity_factor * (vyj - vyi);
            az += viscosity_factor * (vzj - vzi);
        }
    }
    
    // Boundary forces
    for (int k = 0; k < num_boundary_particles; k++) {
        float xk = boundary_positions[3*k];
        float yk = boundary_positions[3*k + 1];
        float zk = boundary_positions[3*k + 2];
        
        float dx = xi - xk;
        float dy = yi - yk;
        float dz = zi - zk;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        
        if (r <= support_radius && r > 1e-9f) {
            float massk = boundary_masses[k];
            
            float grad_x, grad_y, grad_z;
            cubic_kernel_grad(dx, dy, dz, support_radius, &grad_x, &grad_y, &grad_z);
            
            float pressure_factor = massk * pi / (rhoi * rhoi);
            ax -= pressure_factor * grad_x;
            ay -= pressure_factor * grad_y;
            az -= pressure_factor * grad_z;
        }
    }
    
    accelerations[3*i] = ax;
    accelerations[3*i + 1] = ay;
    accelerations[3*i + 2] = az;
}

// Particle integration kernel
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
    
    float gx = gravity[0];
    float gy = gravity[1];
    float gz = gravity[2];
    
    float ax = accelerations[3*i] + gx;
    float ay = accelerations[3*i + 1] + gy;
    float az = accelerations[3*i + 2] + gz;
    
    // Update velocity (symplectic Euler)
    velocities[3*i] += dt * ax;
    velocities[3*i + 1] += dt * ay;
    velocities[3*i + 2] += dt * az;
    
    // Update position
    positions[3*i] += dt * velocities[3*i];
    positions[3*i + 1] += dt * velocities[3*i + 1];
    positions[3*i + 2] += dt * velocities[3*i + 2];
}

// Boundary collision enforcement kernel
__global__ void enforce_boundary_collisions_kernel(
    float* positions,
    float* velocities,
    int num_particles,
    float domain_min_x, float domain_max_x,
    float domain_min_y, float domain_max_y,
    float domain_min_z, float domain_max_z,
    float damping_factor
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float& x = positions[3*i];
    float& y = positions[3*i + 1];
    float& z = positions[3*i + 2];
    
    float& vx = velocities[3*i];
    float& vy = velocities[3*i + 1];
    float& vz = velocities[3*i + 2];
    
    // X boundaries
    if (x < domain_min_x) {
        x = domain_min_x;
        if (vx < 0.0f) vx = -vx * damping_factor;
    } else if (x > domain_max_x) {
        x = domain_max_x;
        if (vx > 0.0f) vx = -vx * damping_factor;
    }
    
    // Y boundaries
    if (y < domain_min_y) {
        y = domain_min_y;
        if (vy < 0.0f) vy = -vy * damping_factor;
    } else if (y > domain_max_y) {
        y = domain_max_y;
        if (vy > 0.0f) vy = -vy * damping_factor;
    }
    
    // Z boundaries
    if (z < domain_min_z) {
        z = domain_min_z;
        if (vz < 0.0f) vz = -vz * damping_factor;
    } else if (z > domain_max_z) {
        z = domain_max_z;
        if (vz > 0.0f) vz = -vz * damping_factor;
    }
}

// C interface functions
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
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((num_particles + blockSize.x - 1) / blockSize.x);
    
    compute_density_kernel<<<gridSize, blockSize>>>(
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
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((num_particles + blockSize.x - 1) / blockSize.x);
    
    compute_pressure_kernel<<<gridSize, blockSize>>>(
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
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((num_particles + blockSize.x - 1) / blockSize.x);
    
    compute_forces_kernel<<<gridSize, blockSize>>>(
        positions, velocities, densities, pressures, masses,
        boundary_positions, boundary_masses, accelerations,
        num_particles, num_boundary_particles,
        support_radius, viscosity
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
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((num_particles + blockSize.x - 1) / blockSize.x);
    
    integrate_particles_kernel<<<gridSize, blockSize>>>(
        positions, velocities, accelerations, gravity,
        num_particles, dt
    );
    
    cudaDeviceSynchronize();
}

void cuda_enforce_boundary_collisions(
    float* positions,
    float* velocities,
    int num_particles,
    float domain_min_x, float domain_max_x,
    float domain_min_y, float domain_max_y,
    float domain_min_z, float domain_max_z,
    float damping_factor
) {
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((num_particles + blockSize.x - 1) / blockSize.x);
    
    enforce_boundary_collisions_kernel<<<gridSize, blockSize>>>(
        positions, velocities, num_particles,
        domain_min_x, domain_max_x,
        domain_min_y, domain_max_y,
        domain_min_z, domain_max_z,
        damping_factor
    );
    
    cudaDeviceSynchronize();
}

} // extern "C"
