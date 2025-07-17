#!/usr/bin/env python3
"""
Performance Analysis and Visualization for SPH Simulation
Generates publication-ready plots from benchmark data
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SPHPerformanceAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load benchmark results from JSON file"""
        results_file = self.results_dir / "benchmark_results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"No benchmark results found at {results_file}")
            
        with open(results_file, 'r') as f:
            self.raw_data = json.load(f)
            
        # Convert to DataFrame for easier analysis
        self.df = pd.json_normalize(self.raw_data)
        
        # Load system info
        system_file = self.results_dir / "system_info.json"
        if system_file.exists():
            with open(system_file, 'r') as f:
                self.system_info = json.load(f)
        else:
            self.system_info = {}
            
    def create_performance_comparison(self):
        """Create comprehensive performance comparison plots"""
        
        if self.df.empty:
            print("No data to plot")
            return
            
        # Group by configuration and mode
        grouped = self.df.groupby(['config.mode', 'config.particle_scale']).agg({
            'avg_frame_time_ms': ['mean', 'std'],
            'particle_count': 'mean'
        }).round(3)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Frame Time vs Particle Scale
        openmp_data = self.df[self.df['config.mode'] == 'openmp']
        cuda_data = self.df[self.df['config.mode'] == 'cuda']
        
        if not openmp_data.empty:
            openmp_grouped = openmp_data.groupby('config.particle_scale')['avg_frame_time_ms'].agg(['mean', 'std'])
            ax1.errorbar(openmp_grouped.index, openmp_grouped['mean'], 
                        yerr=openmp_grouped['std'], label='OpenMP', 
                        marker='o', linewidth=2, markersize=8)
                        
        if not cuda_data.empty:
            cuda_grouped = cuda_data.groupby('config.particle_scale')['avg_frame_time_ms'].agg(['mean', 'std'])
            ax1.errorbar(cuda_grouped.index, cuda_grouped['mean'], 
                        yerr=cuda_grouped['std'], label='CUDA', 
                        marker='s', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Particle Scale Factor')
        ax1.set_ylabel('Average Frame Time (ms)')
        ax1.set_title('Performance Scaling: OpenMP vs CUDA')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Speedup Analysis
        if not openmp_data.empty and not cuda_data.empty:
            speedup_data = []
            scales = set(openmp_data['config.particle_scale']).intersection(
                set(cuda_data['config.particle_scale']))
            
            for scale in sorted(scales):
                openmp_time = openmp_data[openmp_data['config.particle_scale'] == scale]['avg_frame_time_ms'].mean()
                cuda_time = cuda_data[cuda_data['config.particle_scale'] == scale]['avg_frame_time_ms'].mean()
                if cuda_time > 0:
                    speedup_data.append((scale, openmp_time / cuda_time))
            
            if speedup_data:
                scales, speedups = zip(*speedup_data)
                ax2.plot(scales, speedups, 'ro-', linewidth=3, markersize=10)
                ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='No speedup')
                ax2.set_xlabel('Particle Scale Factor')
                ax2.set_ylabel('CUDA Speedup Factor')
                ax2.set_title('CUDA Speedup vs Problem Size')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Add speedup annotations
                for scale, speedup in speedup_data:
                    ax2.annotate(f'{speedup:.1f}x', (scale, speedup), 
                               textcoords="offset points", xytext=(0,10), ha='center')
        
        # Plot 3: Throughput (Particles per second)
        for mode in ['openmp', 'cuda']:
            mode_data = self.df[self.df['config.mode'] == mode]
            if not mode_data.empty:
                # Calculate throughput
                mode_data_copy = mode_data.copy()
                mode_data_copy['throughput'] = mode_data_copy['particle_count'] / (mode_data_copy['avg_frame_time_ms'] / 1000)
                
                grouped_throughput = mode_data_copy.groupby('config.particle_scale')['throughput'].agg(['mean', 'std'])
                ax3.errorbar(grouped_throughput.index, grouped_throughput['mean'], 
                           yerr=grouped_throughput['std'], label=mode.upper(), 
                           marker='o' if mode == 'openmp' else 's', linewidth=2, markersize=8)
        
        ax3.set_xlabel('Particle Scale Factor')
        ax3.set_ylabel('Throughput (Particles/second)')
        ax3.set_title('Computational Throughput Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Plot 4: Efficiency (Performance per hardware unit)
        # This would need hardware-specific information
        cpu_cores = self.system_info.get('cpu_cores', 8)  # Default assumption
        
        for mode in ['openmp', 'cuda']:
            mode_data = self.df[self.df['config.mode'] == mode]
            if not mode_data.empty:
                mode_data_copy = mode_data.copy()
                if mode == 'openmp':
                    # Performance per CPU core
                    mode_data_copy['efficiency'] = mode_data_copy['particle_count'] / (mode_data_copy['avg_frame_time_ms'] * cpu_cores)
                else:
                    # Performance per 100 CUDA cores (normalized)
                    mode_data_copy['efficiency'] = mode_data_copy['particle_count'] / (mode_data_copy['avg_frame_time_ms'] * 0.01)
                
                grouped_eff = mode_data_copy.groupby('config.particle_scale')['efficiency'].agg(['mean', 'std'])
                ax4.errorbar(grouped_eff.index, grouped_eff['mean'], 
                           yerr=grouped_eff['std'], label=f'{mode.upper()}', 
                           marker='o' if mode == 'openmp' else 's', linewidth=2, markersize=8)
        
        ax4.set_xlabel('Particle Scale Factor')
        ax4.set_ylabel('Efficiency (particles/ms/unit)')
        ax4.set_title('Hardware Efficiency Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add overall title with system info
        cpu_model = self.system_info.get('cpu_model', 'Unknown CPU')
        gpu_model = self.system_info.get('gpu_model', 'Unknown GPU')
        fig.suptitle(f'SPH Simulation Performance Analysis\n{cpu_model} vs {gpu_model}', 
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive analysis plot saved to {self.plots_dir}/comprehensive_analysis.png")
        
    def create_scaling_analysis(self):
        """Create detailed scaling analysis"""
        
        if self.df.empty:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Strong scaling analysis (fixed problem size, varying resources)
        # This would need data with different thread counts for OpenMP
        
        # Weak scaling analysis (scaling problem size)
        for mode in ['openmp', 'cuda']:
            mode_data = self.df[self.df['config.mode'] == mode]
            if not mode_data.empty:
                # Calculate particles vs frame time
                ax1.scatter(mode_data['particle_count'], mode_data['avg_frame_time_ms'], 
                           label=mode.upper(), alpha=0.7, s=60)
        
        ax1.set_xlabel('Number of Particles')
        ax1.set_ylabel('Frame Time (ms)')
        ax1.set_title('Weak Scaling: Frame Time vs Problem Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Memory scaling (if available)
        # This would need memory usage data
        ax2.text(0.5, 0.5, 'Memory scaling analysis\nwould require memory\nusage measurements', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Memory Usage Scaling')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "scaling_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Scaling analysis plot saved to {self.plots_dir}/scaling_analysis.png")
        
    def generate_summary_report(self):
        """Generate text summary report"""
        
        report_file = self.results_dir / "reports" / "performance_summary.txt"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write("SPH Simulation Performance Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # System information
            f.write("System Configuration:\n")
            f.write(f"CPU: {self.system_info.get('cpu_model', 'Unknown')}\n")
            f.write(f"GPU: {self.system_info.get('gpu_model', 'Unknown')}\n")
            f.write(f"CPU Cores: {self.system_info.get('cpu_cores', 'Unknown')}\n")
            f.write(f"GPU Memory: {self.system_info.get('gpu_memory_mb', 'Unknown')} MB\n")
            f.write(f"Timestamp: {self.system_info.get('timestamp', 'Unknown')}\n\n")
            
            # Performance summary
            f.write("Performance Summary:\n")
            f.write("-" * 20 + "\n")
            
            if not self.df.empty:
                # Calculate overall statistics
                openmp_data = self.df[self.df['config.mode'] == 'openmp']
                cuda_data = self.df[self.df['config.mode'] == 'cuda']
                
                if not openmp_data.empty:
                    openmp_avg = openmp_data['avg_frame_time_ms'].mean()
                    f.write(f"OpenMP Average Frame Time: {openmp_avg:.2f} ms\n")
                    
                if not cuda_data.empty:
                    cuda_avg = cuda_data['avg_frame_time_ms'].mean()
                    f.write(f"CUDA Average Frame Time: {cuda_avg:.2f} ms\n")
                    
                if not openmp_data.empty and not cuda_data.empty:
                    overall_speedup = openmp_avg / cuda_avg
                    f.write(f"Overall CUDA Speedup: {overall_speedup:.1f}x\n")
                
                f.write(f"\nTotal Benchmark Runs: {len(self.df)}\n")
                f.write(f"Configurations Tested: {len(self.df.groupby(['config.mode', 'config.particle_scale']))}\n")
                
            f.write("\nFiles Generated:\n")
            f.write("- comprehensive_analysis.png: Main performance comparison\n")
            f.write("- scaling_analysis.png: Scaling behavior analysis\n")
            f.write("- benchmark_results.csv: Raw data for further analysis\n")
            
        print(f"Summary report saved to {report_file}")
        
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        try:
            print("Loading benchmark data...")
            self.load_data()
            
            print("Creating performance comparison plots...")
            self.create_performance_comparison()
            
            print("Creating scaling analysis...")
            self.create_scaling_analysis()
            
            print("Generating summary report...")
            self.generate_summary_report()
            
            print(f"\nAnalysis complete! Results available in {self.results_dir}/")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return False
            
        return True

def main():
    parser = argparse.ArgumentParser(description='SPH Performance Analysis')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--plots-only', action='store_true', help='Generate plots only')
    
    args = parser.parse_args()
    
    analyzer = SPHPerformanceAnalyzer(args.results_dir)
    
    if args.plots_only:
        analyzer.load_data()
        analyzer.create_performance_comparison()
        analyzer.create_scaling_analysis()
    else:
        analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
