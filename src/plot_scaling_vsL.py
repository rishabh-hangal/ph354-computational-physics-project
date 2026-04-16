import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# ==========================================
# PUBLICATION FORMATTING (Matched to Paper)
# ==========================================
plt.rcParams.update({
    'font.size': 14,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,       
    'ytick.right': True,     
    'lines.linewidth': 1.5,
    'lines.markersize': 4
})

def plot_paper_scaling_vs_L():
    # =========================================================================
    # CONFIGURATION ZONE
    # =========================================================================
    data_file = "../data/mipt_scaling_master.npz" 
    output_file = "../figures/mipt_loglog_scaling_fig4a.pdf"
    
    # The exact p-values the paper chose to feature in this plot
    target_ps = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 
                 0.16, 0.18, 0.20, 0.22, 0.24, 0.28]
    
    # =========================================================================

    if not os.path.exists(data_file):
        print(f"CRITICAL ERROR: Data file not found at {data_file}")
        return

    data = np.load(data_file)
    L_values = data['L_values']
    p_values = data['p_values']
    
    # Shape is (len(L_values), len(p_values))
    S_mean = data['S_mean'] 
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    markers = ['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h', '8']
    colors = plt.cm.tab20(np.linspace(0, 1, len(target_ps)))
    
    for i, target_p in enumerate(target_ps):
        # Find the closest p_value in your actual data (protects against float math errors)
        p_idx = (np.abs(p_values - target_p)).argmin()
        actual_p = p_values[p_idx]
        
        # EXTRACTING THE COLUMN: 
        # Give me all L values (:) for this specific probability index
        S_line = S_mean[:, p_idx]
        
        # Safety mask for log scale
        safe_S = np.where(S_line > 0, S_line, np.nan)
        marker_style = markers[i % len(markers)]
        
        # SPECIAL CASE: Highlight the critical point (p = 0.16) in bold black
        if abs(actual_p - 0.16) < 0.005:
            ax.plot(L_values, safe_S, marker=marker_style, label=f'$p = {actual_p:.2f}$', 
                    color='black', linewidth=3.5, markerfacecolor='none', zorder=10)
        else:
            ax.plot(L_values, safe_S, marker=marker_style, label=f'$p = {actual_p:.2f}$', 
                    color=colors[i], markerfacecolor='none')

    # 1. Double Logarithmic Scale (Log-Log)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 2. Custom Formatters for BOTH axes (Forces numbers like '10', '50' instead of '10^1')
    def scalar_formatter(val, pos):
        if val >= 1: return f"{int(val)}"
        return f"{val:g}"
            
    ax.xaxis.set_major_formatter(FuncFormatter(scalar_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(scalar_formatter))

    # 3. Grid and Framing
    ax.grid(False) 
    
    # Buffer the limits so curves don't touch the edges
    ax.set_xlim(left=L_values[0]*0.8, right=L_values[-1]*1.2)
    
    # 4. Labels
    ax.set_xlabel('$L$', fontsize=18)
    ax.set_ylabel('$S_A(p; |A| = L/2, L)$', fontsize=18) 
    
    # 5. Legend (Top left, 2 columns)
    ax.legend(loc='upper left', frameon=True, edgecolor='darkgrey', 
              fontsize=12, handlelength=1.5, ncol=2)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"-> Success! Log-Log Scaling vs L safely generated at {output_file}")

if __name__ == '__main__':
    plot_paper_scaling_vs_L()