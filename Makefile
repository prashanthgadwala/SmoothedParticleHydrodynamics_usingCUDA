# SPH Simulation Makefile with Benchmarking Support

# Build directories
BUILD_DIR = build
EXECUTABLE = $(BUILD_DIR)/bin/sph_simulation

# Python interpreter
PYTHON = python3

# Benchmark configuration
BENCHMARK_DIR = benchmark
RESULTS_DIR = results

.PHONY: all build clean benchmark analyze plots report help

# Default target
all: build

# Build the simulation executable
build:
	@echo "Building SPH simulation..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release
	@cd $(BUILD_DIR) && make -j$$(nproc)
	@echo "Build complete: $(EXECUTABLE)"

# Clean build artifacts
clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(RESULTS_DIR)
	@echo "Clean complete"

# Run the simulation normally (with GUI)
run: build
	@echo "Starting SPH simulation..."
	@cd $(BUILD_DIR) && ./bin/sph_simulation

# Check if executable exists
check-executable:
	@if [ ! -f "$(EXECUTABLE)" ]; then \
		echo "Error: Executable not found. Run 'make build' first."; \
		exit 1; \
	fi

# Setup benchmark environment
setup-benchmark:
	@echo "Setting up benchmark environment..."
	@mkdir -p $(RESULTS_DIR)
	@mkdir -p $(RESULTS_DIR)/raw_data
	@mkdir -p $(RESULTS_DIR)/plots
	@mkdir -p $(RESULTS_DIR)/reports

# Check Python dependencies
check-deps:
	@echo "Checking Python dependencies..."
	@$(PYTHON) -c "import matplotlib, numpy, json, csv" 2>/dev/null || \
		(echo "Missing Python dependencies. Install with:" && \
		 echo "  pip install matplotlib numpy" && \
		 exit 1)

# Run automated benchmark suite
benchmark: build check-executable setup-benchmark check-deps
	@echo "Running SPH benchmark suite..."
	@$(PYTHON) benchmark_runner.py $(EXECUTABLE) --output-dir $(RESULTS_DIR)
	@echo "Benchmark complete! Results in $(RESULTS_DIR)/"

# Quick benchmark (fewer configurations, faster)
benchmark-quick: build check-executable setup-benchmark check-deps
	@echo "Running quick SPH benchmark..."
	@$(PYTHON) benchmark_runner.py $(EXECUTABLE) --output-dir $(RESULTS_DIR) --quick
	@echo "Quick benchmark complete! Results in $(RESULTS_DIR)/"

# Analyze existing benchmark results
analyze: check-deps
	@if [ ! -f "$(RESULTS_DIR)/benchmark_results.json" ]; then \
		echo "No benchmark results found. Run 'make benchmark' first."; \
		exit 1; \
	fi
	@echo "Analyzing benchmark results..."
	@$(PYTHON) -c "import json; import sys; sys.path.append('.'); from benchmark_runner import SPHBenchmarkRunner; runner = SPHBenchmarkRunner('dummy', '$(RESULTS_DIR)'); results = json.load(open('$(RESULTS_DIR)/benchmark_results.json')); runner.analyze_results(results)"

# Generate performance plots
plots: analyze
	@echo "Performance plots generated in $(RESULTS_DIR)/plots/"

# Generate comprehensive report
report: plots
	@echo "Generating benchmark report..."
	@$(PYTHON) -c "print('\\n=== SPH Simulation Benchmark Report ===\\n'); import json; data = json.load(open('$(RESULTS_DIR)/system_info.json')); print(f\"System: {data.get('cpu_model', 'Unknown CPU')} + {data.get('gpu_model', 'Unknown GPU')}\"); print(f\"Timestamp: {data.get('timestamp', 'Unknown')}\"); print(f\"\\nResults available in: $(RESULTS_DIR)/\\n\")"

# Test single configuration (for debugging)
test-single: build check-executable
	@echo "Testing single benchmark configuration..."
	@mkdir -p $(RESULTS_DIR)/test
	@$(EXECUTABLE) --headless --benchmark-frames 10 --mode cuda --perf-output $(RESULTS_DIR)/test/test_result.json
	@echo "Test complete. Result in $(RESULTS_DIR)/test/"

# Performance comparison (run both OpenMP and CUDA)
compare: build check-executable setup-benchmark
	@echo "Running performance comparison..."
	@mkdir -p $(RESULTS_DIR)/comparison
	@echo "Testing OpenMP..."
	@$(EXECUTABLE) --headless --benchmark-frames 50 --mode openmp --perf-output $(RESULTS_DIR)/comparison/openmp.json
	@echo "Testing CUDA..."
	@$(EXECUTABLE) --headless --benchmark-frames 50 --mode cuda --perf-output $(RESULTS_DIR)/comparison/cuda.json
	@echo "Comparison complete. Results in $(RESULTS_DIR)/comparison/"

# Show benchmark status
status:
	@echo "=== SPH Benchmark Status ==="
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Executable: $(EXECUTABLE)"
	@if [ -f "$(EXECUTABLE)" ]; then echo "  ✓ Built"; else echo "  ✗ Not built"; fi
	@echo "Results directory: $(RESULTS_DIR)"
	@if [ -d "$(RESULTS_DIR)" ]; then \
		echo "  ✓ Exists"; \
		if [ -f "$(RESULTS_DIR)/benchmark_results.json" ]; then \
			echo "  ✓ Has benchmark data"; \
		else \
			echo "  ✗ No benchmark data"; \
		fi; \
	else \
		echo "  ✗ Not created"; \
	fi

# Show available GPU information
gpu-info:
	@echo "=== GPU Information ==="
	@nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "NVIDIA GPU not detected"

# Help target
help:
	@echo "SPH Simulation Benchmark Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  build          - Build the SPH simulation executable"
	@echo "  run            - Run the simulation with GUI"
	@echo "  clean          - Clean build artifacts and results"
	@echo ""
	@echo "Benchmarking:"
	@echo "  benchmark      - Run complete automated benchmark suite"
	@echo "  benchmark-quick- Run quick benchmark (fewer configurations)"
	@echo "  test-single    - Test single benchmark configuration"
	@echo "  compare        - Quick OpenMP vs CUDA comparison"
	@echo ""
	@echo "Analysis:"
	@echo "  analyze        - Analyze existing benchmark results"
	@echo "  plots          - Generate performance plots"
	@echo "  report         - Generate comprehensive report"
	@echo ""
	@echo "Utilities:"
	@echo "  status         - Show benchmark status"
	@echo "  gpu-info       - Show GPU information"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Example workflow:"
	@echo "  make benchmark    # Run full benchmark suite"
	@echo "  make report       # Generate analysis and plots"
