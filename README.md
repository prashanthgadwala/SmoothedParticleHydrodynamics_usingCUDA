# High-Performance SPH Simulation: OpenMP vs CUDA

A real-time **Smoothed Particle Hydrodynamics (SPH)** fluid simulation implemented in C++ with both **OpenMP** (multi-threaded CPU) and **CUDA** (GPU) backends for performance comparison and analysis.

![SPH Simulation](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)
![OpenMP](https://img.shields.io/badge/OpenMP-4.5%2B-orange.svg)
![CMake](https://img.shields.io/badge/CMake-3.18%2B-red.svg)

## 🚀 Project Overview

This project demonstrates advanced parallel computing techniques by implementing a complete SPH fluid simulation with dual execution paths:

- **OpenMP Implementation**: Multi-threaded CPU simulation utilizing all available cores
- **CUDA Implementation**: GPU-accelerated simulation leveraging thousands of CUDA cores
- **Real-time Performance Comparison**: Interactive switching between implementations with live benchmarking

### Key Features

- ✨ **Complete SPH Physics**: Density estimation, pressure calculation, force computation, and integration
- 🔄 **Dual Implementation**: Side-by-side OpenMP and CUDA performance comparison
- 🎮 **Interactive GUI**: Real-time simulation controls and performance monitoring
- 📊 **Performance Analysis**: Frame time tracking and speedup calculations
- 🏗️ **Modern C++**: C++17 features with professional code architecture
- 🔧 **Cross-platform**: CMake-based build system with automatic dependency management
- ⚖️ **Unified Physics**: Identical mathematical formulations ensure fair performance comparison
- 🎯 **Stability Features**: Advanced clamping and damping for robust simulation

## 🏆 Performance Results

### Typical Performance (1000+ particles)
- **OpenMP (8-16 cores)**: 5-20 ms per frame
- **CUDA (Modern GPU)**: 0.5-3 ms per frame
- **GPU Speedup**: 3-10x faster than multi-threaded CPU

### Why This Comparison Matters
1. **Real-world Relevance**: Compares optimized multi-threaded CPU vs GPU implementations
2. **Architecture Showcase**: Demonstrates understanding of parallel computing paradigms
3. **Practical Application**: Reflects actual HPC decision-making scenarios
4. **Performance Engineering**: Shows optimization skills across different hardware

## 🛠️ Technical Implementation

### SPH Physics Engine
- **Kernel Functions**: Cubic spline kernels with unified OpenMP/CUDA implementation
- **Density Estimation**: Weighted particle mass contributions with boundary handling
- **Pressure Forces**: Tait equation of state with stability clamping
- **Viscosity Forces**: Standard SPH viscosity formulation
- **Boundary Conditions**: Akinci boundary handling with collision detection and damping
- **Time Integration**: Symplectic Euler integration for improved stability
- **Force Clamping**: Acceleration limiting to prevent particle explosion

### Parallel Computing Architecture
```
┌─────────────────┐    ┌─────────────────┐
│   OpenMP CPU    │    │   CUDA GPU      │
│   Implementation│    │   Implementation│
├─────────────────┤    ├─────────────────┤
│ • Multi-threaded│    │ • Kernel-based  │
│ • Cache-friendly│    │ • Massive       │
│ • Loop-level    │    │   parallelism   │
│   parallelism   │    │ • Memory        │
│ • NUMA-aware    │    │   coalescing    │
└─────────────────┘    └─────────────────┘
        │                       │
        └───────────┬───────────┘
                    │
        ┌─────────────────────────┐
        │   Performance Monitor   │
        │   & Comparison Engine   │
        └─────────────────────────┘
```

### Code Architecture
- **Modular Design**: Separate physics, rendering, and UI components
- **Memory Management**: Efficient data structures and GPU memory handling
- **Error Handling**: Comprehensive CUDA error checking and recovery
- **Extensibility**: Plugin-ready architecture for additional features

## 📋 Requirements

### System Requirements
- **OS**: Linux, macOS, or Windows
- **GPU**: NVIDIA GPU with compute capability 6.0+ (Maxwell architecture or newer)
- **RAM**: 4GB+ recommended for large particle counts

### Software Dependencies
- **C++ Compiler**: GCC 9+, Clang 10+, or MSVC 2019+
- **CUDA Toolkit**: 11.0 or later
- **CMake**: 3.18 or later
- **OpenMP**: 4.5 or later (usually included with compiler)

### Graphics Libraries (included via CMake)
- **OpenGL**: For rendering and visualization
- **ImGui**: For interactive user interface
- **Eigen3**: For linear algebra operations

## 🔧 Building and Running

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/SmoothedParticleHydrodynamics_usingCUDA.git
cd SmoothedParticleHydrodynamics_usingCUDA

# Create build directory
mkdir build && cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run the simulation
./bin/sph_simulation
```

### Build Options
```bash
# Build with specific options
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DVISLAB_USE_CUDA=ON \
  -DVISLAB_USE_OPENMP=ON \
  -DVISLAB_BUILD_TESTS=ON
```

### Runtime Controls
- **Simulation Mode**: Toggle between OpenMP, CUDA, or comparison mode
- **Physics Parameters**: Real-time adjustment of stiffness, viscosity, exponent, and damping
- **Time Step Control**: Dynamic time step modification for stability vs performance
- **Boundary Settings**: Adjustable domain damping and collision parameters
- **Visualization**: Real-time particle rendering with pressure-based coloring
- **Performance Monitoring**: Live frame time and speedup statistics

## 🎯 Usage Examples

### Basic Simulation
1. Launch the application
2. Select simulation mode (OpenMP/CUDA/Compare)
3. Adjust particle count and physics parameters
4. Click "Start Simulation" to begin
5. Monitor performance metrics in real-time

### Performance Benchmarking
```cpp
// Example: Programmatic performance testing
SPHSimulation sim;
sim.setMode(SimulationMode::COMPARE_BOTH);
sim.setParticleCount(2000);
sim.runBenchmark(100); // Run 100 frames
auto results = sim.getPerformanceResults();
```

## 📊 Project Structure

```
SmoothedParticleHydrodynamics_usingCUDA/
├── src/
│   ├── SPH/
│   │   ├── main.cpp                 # Main application and OpenMP simulation
│   │   ├── cuda_sph_simulation.hpp  # CUDA simulation interface
│   │   ├── cuda_sph_simulation.cpp  # CUDA memory management and orchestration
│   │   ├── sph_kernels.cu          # CUDA kernel implementations
│   │   └── CMakeLists.txt          # SPH build configuration
│   └── common/                     # Shared utilities and UI framework
├── vislab/                         # Graphics and visualization framework
├── CMakeLists.txt                  # Main build configuration
├── CUDA_Performance_Guide.md       # Detailed performance analysis
└── README.md                       # This file
```

## 🔬 Scientific Background

### Smoothed Particle Hydrodynamics (SPH)
SPH is a computational method for simulating fluid dynamics by discretizing fluid into particles. Each particle carries physical properties (mass, density, pressure) and interacts with neighbors through kernel functions.

### Key Equations
- **Density**: `ρᵢ = Σⱼ mⱼ W(rᵢⱼ, h)`
- **Pressure**: `P = k((ρ/ρ₀)^γ - 1)`
- **Force**: `Fᵢ = -Σⱼ mⱼ(Pᵢ/ρᵢ² + Pⱼ/ρⱼ²)∇W(rᵢⱼ, h)`

### Applications
- **Visual Effects**: Movie and game fluid simulations
- **Engineering**: Coastal engineering, dam break analysis
- **Research**: Astrophysics, geophysics, biomechanics

## 🚀 Performance Optimizations

### Recent Improvements
- **Unified Kernel Implementation**: Fixed mathematical inconsistencies between OpenMP and CUDA
- **Stability Enhancements**: Added pressure/density clamping and acceleration limiting
- **Mass Calculation**: Unified mass distribution for fair performance comparison
- **Integration Method**: Symplectic Euler for improved numerical stability
- **Boundary Handling**: Enhanced collision detection with proper velocity reflection

### CPU Optimizations (OpenMP)
- **Thread-local Storage**: Minimize false sharing
- **Cache Optimization**: Structure-of-Arrays layout
- **Load Balancing**: Dynamic work distribution
- **Spatial Data Structures**: Efficient nearest neighbor search

### GPU Optimizations (CUDA)
- **Memory Coalescing**: Optimal memory access patterns
- **Kernel Synchronization**: Proper device synchronization
- **Block Size Tuning**: 256 threads per block for optimal occupancy
- **Unified Memory Management**: Efficient host-device data transfer

## 🎓 Educational Value

This project demonstrates proficiency in:
- **Parallel Programming**: OpenMP and CUDA expertise
- **Scientific Computing**: Numerical methods and physics simulation
- **Performance Engineering**: Optimization and benchmarking
- **Software Architecture**: Modern C++ design patterns
- **Graphics Programming**: Real-time rendering and visualization
- **Build Systems**: CMake and cross-platform development

## 📈 Future Enhancements

- [ ] **Advanced Kernels**: Wendland and spiky kernels
- [ ] **Surface Reconstruction**: Marching cubes mesh generation
- [ ] **Multi-GPU Support**: Distributed computing across multiple GPUs
- [ ] **Adaptive Timesteps**: Dynamic stability control
- [ ] **Fluid-Structure Interaction**: Rigid body coupling
- [ ] **Turbulence Modeling**: Large eddy simulation (LES)

## 🤝 Contributing

This project is part of my academic portfolio. Feel free to:
- Report issues or suggest improvements
- Fork for educational purposes
- Discuss implementation details
- Suggest performance optimizations

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This project was developed as part of advanced coursework in High-End Simulation in Practice, demonstrating practical skills in parallel computing, scientific simulation, and performance optimization.*