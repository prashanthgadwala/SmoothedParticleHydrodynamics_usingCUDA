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
    # Save with particle count in filename
    particle_count = int(df['Particles'].iloc[0])
    plt.savefig(f'sph_performance_{particle_count}_particles.png', dpi=300, bbox_inches='tight')
    plt.show()

# def get_latest_performance_csv():
#     import glob
#     import re
#     files = glob.glob(os.path.join('output', 'sph_performance_*.csv'))
#     if not files:
#         raise FileNotFoundError('No performance CSV files found in output directory.')
#     # Extract frame numbers
#     def extract_frame(f):
#         m = re.search(r'sph_performance_(\d+)\.csv$', f)  # <-- fixed regex
#         return int(m.group(1)) if m else -1
#     files = [(f, extract_frame(f)) for f in files]
#     files = [f for f in files if f[1] != -1]
#     if not files:
#         raise FileNotFoundError('No valid performance CSV files found.')
#     latest = max(files, key=lambda x: x[1])[0]
#     return os.path.basename(latest)

# Usage: automatically plot the latest file
plot_performance_analysis('sph_performance_2500.csv')