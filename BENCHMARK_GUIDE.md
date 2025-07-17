# SPH Simulation Benchmarking Guide

This guide explains how to run performance benchmarks and generate analysis for your SPH simulation project.

## Quick Start

### 1. Build and Test
```bash
# Build the simulation
make build

# Test that everything works
make run
```

### 2. Run Benchmarks
```bash
# Run complete benchmark suite (recommended)
make benchmark

# Or run quick benchmark (faster, fewer configurations)
make benchmark-quick
```

### 3. Generate Analysis
```bash
# Analyze results and generate plots
make report
```

## What Gets Measured

### Performance Metrics
- **Frame Time**: Time per simulation step (ms)
- **Throughput**: Particles processed per second
- **Speedup**: CUDA performance vs OpenMP ratio
- **Efficiency**: Performance per hardware unit

### Test Configurations
- **Particle Scales**: 0.5x, 1.0x, 1.5x, 2.0x (relative to base configuration)
- **Support Radii**: 0.15, 0.2, 0.25 (affects neighbor search)
- **Modes**: OpenMP (CPU) vs CUDA (GPU)
- **Multiple Runs**: Each configuration tested 3 times for statistical validity

## Output Files

### Results Directory Structure
```
results/
├── benchmark_results.json     # Raw benchmark data
├── benchmark_results.csv      # CSV format for analysis
├── system_info.json          # Hardware specifications
├── plots/
│   ├── comprehensive_analysis.png  # Main performance plots
│   └── scaling_analysis.png        # Scaling behavior
└── reports/
    └── performance_summary.txt     # Text summary report
```

### Key Plots Generated
1. **Performance Comparison**: OpenMP vs CUDA frame times
2. **Speedup Analysis**: CUDA advantage across problem sizes
3. **Throughput Comparison**: Computational efficiency
4. **Hardware Efficiency**: Performance per CPU core / GPU unit

## Manual Benchmark Execution

### Single Test
```bash
# Test CUDA with specific parameters
./build/bin/sph_simulation --headless --benchmark-frames 50 --mode cuda --particle-scale 1.0

# Test OpenMP
./build/bin/sph_simulation --headless --benchmark-frames 50 --mode openmp --particle-scale 1.0
```

### Custom Python Script
```bash
# Run automated benchmark with custom parameters
python3 benchmark_runner.py ./build/bin/sph_simulation --output-dir my_results
```

## Command Line Options

Your simulation now supports these command-line arguments:
- `--headless`: Run without GUI (for automated testing)
- `--benchmark-frames N`: Run N frames then exit
- `--mode MODE`: Force OpenMP or CUDA mode
- `--particle-scale SCALE`: Scale particle count (0.5 = half, 2.0 = double)
- `--support-radius RADIUS`: Override support radius
- `--perf-output FILE`: Export performance data to JSON file

## Interpreting Results

### Expected CUDA Advantages
- **Large Problem Sizes**: 3-10x speedup for 2000+ particles
- **Parallel Phases**: Best speedup in density/pressure computation
- **Memory Bandwidth**: GPU excels with high particle density

### Expected OpenMP Advantages
- **Small Problems**: Lower overhead for <500 particles
- **Cache Efficiency**: Better performance with sparse neighborhoods
- **Development**: Easier debugging and profiling

### Key Metrics for Presentation
1. **Crossover Point**: Particle count where CUDA becomes beneficial
2. **Maximum Speedup**: Best CUDA performance ratio achieved
3. **Scaling Efficiency**: How performance degrades with problem size
4. **Memory Usage**: GPU memory requirements vs problem size

## Troubleshooting

### Build Issues
```bash
# Check CUDA availability
make gpu-info

# Clean and rebuild
make clean
make build
```

### Benchmark Issues
```bash
# Check benchmark status
make status

# Test single configuration
make test-single

# Quick comparison
make compare
```

### Missing Dependencies
```bash
# Install Python packages
pip install matplotlib numpy pandas

# For seaborn (optional, for better plots)
pip install seaborn
```

## For Your Video Presentation

### 5-Minute Structure
1. **Problem Overview** (30s): SPH simulation challenges
2. **Implementation** (60s): OpenMP vs CUDA approaches
3. **Benchmark Results** (150s): Show key performance plots
4. **Analysis** (90s): Explain speedup patterns and bottlenecks
5. **Conclusions** (30s): When to use GPU vs CPU

### Key Plots to Show
- Performance comparison bar chart
- Speedup line graph
- Scaling efficiency plot

### Talking Points
- GPU memory bandwidth advantage
- CPU cache efficiency for small problems
- Parallel algorithm design considerations
- Real-world performance implications

## Advanced Usage

### Custom Configurations
Edit `benchmark_runner.py` to add new test configurations:
```python
configs.append({
    "name": "Custom Test",
    "particle_scale": 1.2,
    "support_radius": 0.3,
    "frames": 75,
    "mode": "cuda"
})
```

### Integration with SLURM
```bash
# Submit benchmark job to cluster
sbatch run_gpu_simulation.sbatch make benchmark
```

### Data Export
Results are automatically exported in multiple formats:
- JSON for programmatic access
- CSV for spreadsheet analysis
- PNG plots for presentations

This benchmarking system works entirely with your existing simulation code without modifying the core physics or rendering logic.
