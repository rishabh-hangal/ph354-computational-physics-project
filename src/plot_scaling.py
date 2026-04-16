import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# ==========================================
# PUBLICATION FORMATTING
# ==========================================
plt.rcParams.update({
    'font.size': 16,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    # Turn on ticks for the top and right borders
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 1.5,
    'lines.markersize': 5
})

def plot_paper_replica():
    # =========================================================================
    # CONFIGURATION ZONE
    # =========================================================================
    data_file = "../data/mipt_scaling_master.npz" 
    output_file = "../figures/mipt_logS_vs_p_paper_format.pdf"
    
    print(f"Loading scaling data from {data_file}...")
    
    if not os.path.exists(data_file):
        print(f"CRITICAL ERROR: Data file not found at {data_file}")
        return

    data = np.load(data_file)
    L_values = data['L_values']
    p_values = data['p_values']
    S_mean = data['S_mean']
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    # We will loop through these to match the paper's varying marker styles
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']
    
    # Using a similar muted colormap to the paper (e.g., Seaborn deep or similar)
    colors = plt.cm.tab10(np.linspace(0, 1, len(L_values)))
    
    for i, L in enumerate(L_values):
        S_line = S_mean[i, :]
        marker_style = markers[i % len(markers)]
        
        ax.plot(p_values, S_line, marker=marker_style, label=f'$L={L}$', 
                color=colors[i], markerfacecolor='none') # 'none' makes markers hollow like the paper

    # 1. Logarithmic Y-Axis
    ax.set_yscale('log')
    
    # 2. Fix the Cropping! Set bottom low enough to catch your data, top to 150
    ax.set_ylim(bottom=0.01, top=1000)
    
    # The paper's x-axis spans exactly 0 to 1
    ax.set_xlim(left=-0.02, right=1.02) 

    # 3. Custom formatter to write '0.5' instead of '5x10^-1'
    def log_formatter(y, pos):
        if y >= 1:
            return f"{int(y)}"
        else:
            return f"{y:g}"
            
    ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))

    # 4. Grid Formatting: Vertical dotted lines ONLY
    ax.grid(False) 
    ax.xaxis.grid(True, linestyle=':', alpha=0.6, color='black')

    # 5. Labels
    ax.set_xlabel('$p$', fontsize=18)
    # Ensure this matches your actual measurement (e.g., L/2 or L/4)
    ax.set_ylabel('$S_A(L/2)$', fontsize=18) 
    
    # 6. Legend inside the plot, top right, with a grey border
    ax.legend(loc='upper right', frameon=True, edgecolor='darkgrey', 
              fontsize=14, handlelength=1.5)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"-> Success! Plot safely generated at {output_file}")

if __name__ == '__main__':
    plot_paper_replica()