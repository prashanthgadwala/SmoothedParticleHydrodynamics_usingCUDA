#!/bin/bash
# Setup script for SPH benchmarking environment

echo "=== SPH Benchmark Environment Setup ==="

# Check if we're in the right directory
if [ ! -f "src/SPH/main.cpp" ]; then
    echo "Error: Please run this script from the SPH project root directory"
    exit 1
fi

echo "✓ Found SPH project files"

# Check build tools
echo "Checking build tools..."
if command -v cmake &> /dev/null; then
    echo "✓ CMake found: $(cmake --version | head -n1)"
else
    echo "✗ CMake not found - install with: sudo apt install cmake"
    exit 1
fi

if command -v make &> /dev/null; then
    echo "✓ Make found"
else
    echo "✗ Make not found - install with: sudo apt install build-essential"
    exit 1
fi

# Check CUDA
echo "Checking CUDA..."
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA compiler found: $(nvcc --version | grep release)"
else
    echo "⚠ CUDA compiler not found - GPU benchmarks will not work"
fi

if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA driver found:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1
else
    echo "⚠ NVIDIA driver not found - GPU benchmarks will not work"
fi

# Check Python and packages
echo "Checking Python environment..."
if command -v python3 &> /dev/null; then
    echo "✓ Python3 found: $(python3 --version)"
else
    echo "✗ Python3 not found - install with: sudo apt install python3"
    exit 1
fi

# Check Python packages
echo "Checking Python packages..."
python3 -c "import matplotlib" 2>/dev/null && echo "✓ matplotlib" || echo "✗ matplotlib - install with: pip install matplotlib"
python3 -c "import numpy" 2>/dev/null && echo "✓ numpy" || echo "✗ numpy - install with: pip install numpy"
python3 -c "import pandas" 2>/dev/null && echo "✓ pandas" || echo "✗ pandas - install with: pip install pandas"
python3 -c "import json" 2>/dev/null && echo "✓ json (built-in)"
python3 -c "import csv" 2>/dev/null && echo "✓ csv (built-in)"

# Optional packages
python3 -c "import seaborn" 2>/dev/null && echo "✓ seaborn (optional)" || echo "○ seaborn (optional) - install with: pip install seaborn"

# Create directories
echo "Creating benchmark directories..."
mkdir -p results/raw_data
mkdir -p results/plots
mkdir -p results/reports
echo "✓ Created results directories"

# Test build
echo "Testing build..."
if make build; then
    echo "✓ Build successful"
else
    echo "✗ Build failed - check CMake configuration"
    exit 1
fi

# Test executable
if [ -f "build/bin/sph_simulation" ]; then
    echo "✓ Executable created: build/bin/sph_simulation"
else
    echo "✗ Executable not found"
    exit 1
fi

echo ""
echo "=== Setup Complete ==="
echo "Your environment is ready for SPH benchmarking!"
echo ""
echo "Quick start:"
echo "  make benchmark      # Run full benchmark suite"
echo "  make report         # Generate analysis and plots"
echo ""
echo "For more options: make help"
