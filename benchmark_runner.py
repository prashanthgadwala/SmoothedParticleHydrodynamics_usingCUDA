#!/usr/bin/env python3
"""
SPH Simulation Benchmark Runner
Automates performance testing of OpenMP vs CUDA SPH simulation
"""

import os
import sys
import json
import csv
import time
import subprocess
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class SPHBenchmarkRunner:
    def __init__(self, executable_path, output_dir="results"):
        self.executable_path = Path(executable_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "raw_data").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        self.results = []
        
    def get_system_info(self):
        """Collect system information for benchmarking context"""
        info = {
            "timestamp": datetime.now().isoformat(),
            "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown"
        }
        
        # Get CPU info
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
                for line in cpu_info.split('\n'):
                    if 'model name' in line:
                        info['cpu_model'] = line.split(':')[1].strip()
                        break
                    if 'cpu cores' in line:
                        info['cpu_cores'] = int(line.split(':')[1].strip())
        except:
            info['cpu_model'] = "unknown"
            info['cpu_cores'] = "unknown"
            
        # Get GPU info
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_line = result.stdout.strip().split('\n')[0]
                gpu_name, gpu_memory = gpu_line.split(', ')
                info['gpu_model'] = gpu_name.strip()
                info['gpu_memory_mb'] = int(gpu_memory.strip())
        except:
            info['gpu_model'] = "unknown"
            info['gpu_memory_mb'] = "unknown"
            
        return info
    
    def run_benchmark_configuration(self, config, runs=5):
        """Run a single benchmark configuration multiple times"""
        print(f"Running benchmark: {config['name']}")
        
        run_results = []
        
        for run in range(runs):
            print(f"  Run {run + 1}/{runs}")
            
            # Build command line arguments
            cmd = [str(self.executable_path)]
            
            # Add configuration parameters
            if 'particle_scale' in config:
                cmd.extend(['--particle-scale', str(config['particle_scale'])])
            if 'support_radius' in config:
                cmd.extend(['--support-radius', str(config['support_radius'])])
            if 'frames' in config:
                cmd.extend(['--benchmark-frames', str(config['frames'])])
            if 'mode' in config:
                cmd.extend(['--mode', config['mode']])
                
            # Add headless mode for automated testing
            cmd.append('--headless')
            
            # Add performance output flag
            cmd.extend(['--perf-output', str(self.output_dir / "raw_data" / f"run_{run}.json")])
            
            try:
                # Run the simulation
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # Parse the performance output
                    perf_file = self.output_dir / "raw_data" / f"run_{run}.json"
                    if perf_file.exists():
                        with open(perf_file, 'r') as f:
                            perf_data = json.load(f)
                            perf_data['config'] = config
                            perf_data['run'] = run
                            run_results.append(perf_data)
                else:
                    print(f"    Error in run {run}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"    Timeout in run {run}")
            except Exception as e:
                print(f"    Exception in run {run}: {e}")
                
        return run_results
    
    def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        
        # Define benchmark configurations
        configs = [
            {
                "name": "Small Scale - OpenMP",
                "particle_scale": 0.5,
                "support_radius": 0.25,
                "frames": 100,
                "mode": "openmp"
            },
            {
                "name": "Small Scale - CUDA",
                "particle_scale": 0.5,
                "support_radius": 0.25,
                "frames": 100,
                "mode": "cuda"
            },
            {
                "name": "Medium Scale - OpenMP",
                "particle_scale": 1.0,
                "support_radius": 0.25,
                "frames": 100,
                "mode": "openmp"
            },
            {
                "name": "Medium Scale - CUDA",
                "particle_scale": 1.0,
                "support_radius": 0.25,
                "frames": 100,
                "mode": "cuda"
            },
            {
                "name": "Large Scale - OpenMP",
                "particle_scale": 1.5,
                "support_radius": 0.2,
                "frames": 50,
                "mode": "openmp"
            },
            {
                "name": "Large Scale - CUDA",
                "particle_scale": 1.5,
                "support_radius": 0.2,
                "frames": 50,
                "mode": "cuda"
            },
            {
                "name": "Very Large Scale - CUDA Only",
                "particle_scale": 2.0,
                "support_radius": 0.15,
                "frames": 25,
                "mode": "cuda"
            }
        ]
        
        system_info = self.get_system_info()
        
        # Save system info
        with open(self.output_dir / "system_info.json", 'w') as f:
            json.dump(system_info, f, indent=2)
            
        print("=== SPH Simulation Benchmark Suite ===")
        print(f"System: {system_info.get('cpu_model', 'unknown')} + {system_info.get('gpu_model', 'unknown')}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Run all configurations
        all_results = []
        for config in configs:
            results = self.run_benchmark_configuration(config, runs=3)
            all_results.extend(results)
            
        # Save all results
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        # Generate CSV for easy analysis
        self.export_csv(all_results)
        
        # Generate analysis
        self.analyze_results(all_results)
        
        print(f"\nBenchmark complete! Results saved to {self.output_dir}")
        
    def export_csv(self, results):
        """Export results to CSV format"""
        csv_file = self.output_dir / "benchmark_results.csv"
        
        with open(csv_file, 'w', newline='') as f:
            if not results:
                return
                
            # Get all possible keys for CSV headers
            headers = set()
            for result in results:
                headers.update(result.keys())
                if 'config' in result:
                    headers.update(result['config'].keys())
            
            headers = sorted(list(headers))
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for result in results:
                row = result.copy()
                if 'config' in result:
                    config = row.pop('config')
                    row.update(config)
                writer.writerow(row)
                
    def analyze_results(self, results):
        """Generate performance analysis and plots"""
        if not results:
            print("No results to analyze")
            return
            
        print("\n=== Performance Analysis ===")
        
        # Group results by configuration
        openmp_results = [r for r in results if r.get('config', {}).get('mode') == 'openmp']
        cuda_results = [r for r in results if r.get('config', {}).get('mode') == 'cuda']
        
        # Calculate average performance
        print("\nAverage Frame Times:")
        
        openmp_by_scale = {}
        cuda_by_scale = {}
        
        for result in openmp_results:
            scale = result.get('config', {}).get('particle_scale', 'unknown')
            if scale not in openmp_by_scale:
                openmp_by_scale[scale] = []
            if 'avg_frame_time_ms' in result:
                openmp_by_scale[scale].append(result['avg_frame_time_ms'])
                
        for result in cuda_results:
            scale = result.get('config', {}).get('particle_scale', 'unknown')
            if scale not in cuda_by_scale:
                cuda_by_scale[scale] = []
            if 'avg_frame_time_ms' in result:
                cuda_by_scale[scale].append(result['avg_frame_time_ms'])
        
        # Print performance summary
        for scale in sorted(set(list(openmp_by_scale.keys()) + list(cuda_by_scale.keys()))):
            print(f"\nScale {scale}:")
            
            if scale in openmp_by_scale and openmp_by_scale[scale]:
                openmp_avg = np.mean(openmp_by_scale[scale])
                openmp_std = np.std(openmp_by_scale[scale])
                print(f"  OpenMP: {openmp_avg:.2f} ± {openmp_std:.2f} ms")
                
            if scale in cuda_by_scale and cuda_by_scale[scale]:
                cuda_avg = np.mean(cuda_by_scale[scale])
                cuda_std = np.std(cuda_by_scale[scale])
                print(f"  CUDA:   {cuda_avg:.2f} ± {cuda_std:.2f} ms")
                
                if scale in openmp_by_scale and openmp_by_scale[scale]:
                    speedup = openmp_avg / cuda_avg
                    print(f"  Speedup: {speedup:.1f}x")
        
        # Generate plots
        self.generate_plots(openmp_by_scale, cuda_by_scale)
        
    def generate_plots(self, openmp_data, cuda_data):
        """Generate performance visualization plots"""
        
        # Performance comparison plot
        scales = sorted(set(list(openmp_data.keys()) + list(cuda_data.keys())))
        openmp_means = []
        cuda_means = []
        openmp_stds = []
        cuda_stds = []
        
        for scale in scales:
            if scale in openmp_data and openmp_data[scale]:
                openmp_means.append(np.mean(openmp_data[scale]))
                openmp_stds.append(np.std(openmp_data[scale]))
            else:
                openmp_means.append(0)
                openmp_stds.append(0)
                
            if scale in cuda_data and cuda_data[scale]:
                cuda_means.append(np.mean(cuda_data[scale]))
                cuda_stds.append(np.std(cuda_data[scale]))
            else:
                cuda_means.append(0)
                cuda_stds.append(0)
        
        # Create performance comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Frame time comparison
        x = np.arange(len(scales))
        width = 0.35
        
        ax1.bar(x - width/2, openmp_means, width, yerr=openmp_stds, label='OpenMP', alpha=0.8)
        ax1.bar(x + width/2, cuda_means, width, yerr=cuda_stds, label='CUDA', alpha=0.8)
        
        ax1.set_xlabel('Particle Scale')
        ax1.set_ylabel('Average Frame Time (ms)')
        ax1.set_title('Performance Comparison: OpenMP vs CUDA')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{s}x' for s in scales])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Speedup plot
        speedups = []
        valid_scales = []
        for i, scale in enumerate(scales):
            if openmp_means[i] > 0 and cuda_means[i] > 0:
                speedups.append(openmp_means[i] / cuda_means[i])
                valid_scales.append(scale)
        
        if speedups:
            ax2.plot(valid_scales, speedups, 'o-', linewidth=2, markersize=8)
            ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='No speedup')
            ax2.set_xlabel('Particle Scale')
            ax2.set_ylabel('CUDA Speedup Factor')
            ax2.set_title('CUDA Speedup vs Problem Size')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to {self.output_dir / 'plots'}")

def main():
    parser = argparse.ArgumentParser(description='SPH Simulation Benchmark Runner')
    parser.add_argument('executable', help='Path to SPH simulation executable')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark with fewer configurations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.executable):
        print(f"Error: Executable not found: {args.executable}")
        sys.exit(1)
    
    runner = SPHBenchmarkRunner(args.executable, args.output_dir)
    runner.run_all_benchmarks()

if __name__ == "__main__":
    main()
