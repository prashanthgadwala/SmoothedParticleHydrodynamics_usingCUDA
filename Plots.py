# performance_plotter.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_performance_analysis(csv_file):
    
    csv_file = os.path.join('output', csv_file)
    # Read data
    df = pd.read_csv(csv_file)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Time comparison
    ax1.plot(df['Frame'], df['OpenMP_ms'], 'b-', label='OpenMP', linewidth=2)
    ax1.plot(df['Frame'], df['CUDA_ms'], 'r-', label='CUDA', linewidth=2)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('OpenMP vs CUDA Performance')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Speedup over time
    ax2.plot(df['Frame'], df['Speedup'], 'g-', linewidth=2)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('CUDA Speedup Over Time')
    ax2.grid(True)
    
    # 3. Performance vs particle count
    ax3.scatter(df['Particles'], df['OpenMP_ms'], alpha=0.6, label='OpenMP')
    ax3.scatter(df['Particles'], df['CUDA_ms'], alpha=0.6, label='CUDA')
    ax3.set_xlabel('Number of Particles')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Performance vs Particle Count')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Efficiency analysis
    efficiency = df['Speedup'] / df['Particles'] * 1000  # Normalize
    ax4.plot(df['Frame'], efficiency, 'm-', linewidth=2)
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Efficiency (Speedup/1K Particles)')
    ax4.set_title('CUDA Efficiency Analysis')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('sph_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Usage
plot_performance_analysis('sph_performance_500.csv')