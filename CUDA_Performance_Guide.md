# OpenMP vs CUDA Performance Comparison for SPH Simulation

This project implements a Smoothed Particle Hydrodynamics (SPH) simulation with both OpenMP (multi-threaded CPU) and CUDA (GPU) implementations for realistic performance comparison.

## Features

- **OpenMP Implementation**: Multi-threaded CPU simulation using all available CPU cores
- **CUDA Implementation**: GPU-accelerated simulation using thousands of CUDA cores
- **Performance Comparison**: Real-time comparison of both implementations
- **Interactive GUI**: Switch between simulation modes during runtime
- **Complete SPH Implementation**: Includes density estimation, pressure calculation, force computation, and integration

## Simulation Modes

1. **OpenMP Only**: Runs simulation on multi-threaded CPU only
2. **CUDA Only**: Runs simulation on GPU only  
3. **Compare Both**: Runs both implementations and shows performance comparison

## Expected Performance Results

For typical SPH simulations with 1000+ particles:

- **OpenMP (8-16 cores)**: Usually 5-20 ms per frame depending on particle count and CPU
- **CUDA (Modern GPU)**: Usually 0.5-3 ms per frame on modern GPUs
- **Expected Speedup**: 3-10x faster on GPU (realistic comparison)

## Why This Comparison is More Meaningful

1. **Fair Comparison**: OpenMP utilizes all CPU cores vs GPU's thousands of cores
2. **Real-world Scenario**: Most high-performance computing uses multi-threading
3. **Architecture Showcase**: Shows the difference between CPU and GPU architectures
4. **Practical Relevance**: Reflects actual HPC decision-making scenarios

## SPH Implementation Details

### Complete Physics
- **Density Estimation**: Using cubic spline kernels
- **Pressure Calculation**: Tait equation of state
- **Force Computation**: Pressure and viscosity forces
- **Akinci Boundary Conditions**: Proper boundary handling

### Optimizations
- Smaller particle size for smoother visuals
- Optimized timesteps for stability
- Memory-efficient data structures

## Building

Make sure you have:
- CUDA Toolkit installed
- Compatible NVIDIA GPU
- CMake 3.18+

The CMakeLists.txt automatically detects CUDA and builds accordingly.
