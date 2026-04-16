import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# ==========================================
# PUBLICATION FORMATTING (Matched to Paper)
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
    'xtick.top': True,       
    'ytick.right': True,     
    'lines.linewidth': 1.5,
    'lines.markersize': 4
})

def plot_halfchain_loglog_page_curve():
    # =========================================================================
    # CONFIGURATION ZONE
    # =========================================================================
    # Point this to your full triangle Page Curve data file
    data_file = "../data/p=0.0,0.05,0.1,0.15,0.2,0.25_mipt_page_curve_L128.npz"  # Update L if necessary
    output_file = "../figures/mipt_loglog_page_curve_fig4b.pdf"
    
    # The exact 14 p-values the paper chose to feature in the legend
    target_ps = [0.00, 0.05, 0.10, 0.15, 0.18, 0.20, 0.25]  # Update if your data has different p-values
    
    # =========================================================================

    print(f"Loading Page Curve data from {data_file}...")
    
    if not os.path.exists(data_file):
        print(f"CRITICAL ERROR: Data file not found at {data_file}")
        return

    data = np.load(data_file)
    p_values = data['p_values']
    cuts = data['cuts']            
    S_mean = data['S_mean_page']   
    
    # 1. DYNAMICALLY SLICE TO HALF-CHAIN ONLY
    # If cuts goes up to L-1, then L is max(cuts) + 1
    L_actual = int(np.max(cuts) + 1)
    
    # Create a mask to only keep data where |A| <= L/2
    half_chain_mask = cuts <= (L_actual // 2)
    
    cuts_half = cuts[half_chain_mask]
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    # Use tab20 to get enough distinct colors for all 14 lines
    colors = plt.cm.tab20(np.linspace(0, 1, len(target_ps)))
    markers = ['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h', '8']
    
    for i, target_p in enumerate(target_ps):
        p_idx = (np.abs(p_values - target_p)).argmin()
        actual_p = p_values[p_idx]
        
        # Apply the mask to the entropy data to chop off the right side of the triangle
        # Apply the mask to chop off the right side of the triangle
        S_line_half = S_mean[p_idx, half_chain_mask]
        
        # SLICE TO REMOVE SPATIAL HALF-STEPS
        # This forces the plot to only look at every other cut (e.g., even bonds)
        cuts_plot = cuts_half[::2]
        S_plot = S_line_half[::2]
        
        # Safety mask for log scale
        safe_S = np.where(S_plot > 0, S_plot, np.nan)
        marker_style = markers[i % len(markers)]
        
        # Highlight the critical point (p = 0.16) in bold black
        if abs(actual_p - 0.16) < 0.005:
            ax.plot(cuts_plot, safe_S, marker=marker_style, label=f'$p = {actual_p:.2f}$', 
                    color='black', linewidth=3.5, markerfacecolor='none', zorder=10)
        else:
            ax.plot(cuts_plot, safe_S, marker=marker_style, label=f'$p = {actual_p:.2f}$', 
                    color=colors[i], markerfacecolor='none')

    # 2. DOUBLE LOGARITHMIC SCALE (Log-Log)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 3. Custom Formatters (Forces numbers like '10', '50' instead of '10^1')
    def scalar_formatter(val, pos):
        if val >= 1: return f"{int(val)}"
        return f"{val:g}"
            
    ax.xaxis.set_major_formatter(FuncFormatter(scalar_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(scalar_formatter))

    # 4. Grid and Framing
    ax.grid(False) 
    
    # Auto-frame the limits to look clean
    ax.set_xlim(left=max(0.8, cuts_half[0]*0.8), right=cuts_half[-1]*1.1)

    # 5. Labels
    ax.set_xlabel('$|A|$', fontsize=18)
    ax.set_ylabel(f'$S_A(p; |A|, L={L_actual})$', fontsize=18) 
    
    # 6. Legend (Top left, 2 columns, exactly like the paper)
    ax.legend(loc='upper left', frameon=True, edgecolor='darkgrey', 
              fontsize=12, handlelength=1.5, ncol=2)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"-> Success! Half-Chain Log-Log Page Curve generated at {output_file}")

if __name__ == '__main__':
    plot_halfchain_loglog_page_curve()