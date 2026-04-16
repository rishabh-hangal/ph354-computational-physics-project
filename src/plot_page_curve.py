import numpy as np
import matplotlib.pyplot as plt
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
    'xtick.top': True,       
    'ytick.right': True,     
    'lines.linewidth': 2.0,
    'lines.markersize': 5
})

def plot_full_page_curve():
    # =========================================================================
    # CONFIGURATION ZONE
    # =========================================================================
    # Point this strictly to your full Page Curve data file
    data_file = "../data/p=0.0,0.05,0.1,0.15,0.2,0.25_mipt_page_curve_L128.npz" 
    output_file = "../figures/mipt_page_curve_triangle.pdf"
    
    # If your file has 50 p-values, plotting all of them is a mess. 
    # Pick a clean subset that spans across the transition (e.g., pc ~ 0.16)
    target_ps = [0.00, 0.05, 0.10, 0.15, 0.18, 0.25, 0.40]
    
    # =========================================================================

    print(f"Loading Page Curve data from {data_file}...")
    
    if not os.path.exists(data_file):
        print(f"CRITICAL ERROR: Data file not found at {data_file}")
        return

    data = np.load(data_file)
    p_values = data['p_values']
    cuts = data['cuts']            # The subsystem sizes |A|
    
    # Expected shape: (len(p_values), len(cuts))
    S_mean = data['S_mean_page']   
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    markers = ['o', 's', 'D', '^', 'v', '<', '>']
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(target_ps)))
    
    global_max_S = 0

    for i, target_p in enumerate(target_ps):
        # Safely find the closest p in your array
        p_idx = (np.abs(p_values - target_p)).argmin()
        actual_p = p_values[p_idx]
        
        # Extract the full triangle line for this probability
        S_line = S_mean[p_idx, :]
        
        # Track the absolute maximum to frame the plot perfectly
        if np.max(S_line) > global_max_S:
            global_max_S = np.max(S_line)
            
        marker_style = markers[i % len(markers)]
        
        # Highlight the critical/near-critical point in red to make it pop
        if abs(actual_p - 0.16) < 0.01:
            ax.plot(cuts, S_line, marker=marker_style, label=f'$p = {actual_p:.2f}$ (Near $p_c$)', 
                    color='#d62728', linewidth=3.0, zorder=10)
        else:
            ax.plot(cuts, S_line, marker=marker_style, label=f'$p = {actual_p:.2f}$', 
                    color=colors[i])

    # 1. Dynamic Framing (Linear Scale)
    # The entropy for a pure state must go to 0 at the boundaries, so bottom is strictly 0.
    # The top is dynamically buffered 15% above your highest volume-law peak.
    ax.set_ylim(bottom=0.0, top=global_max_S * 1.15)
    
    # Frame the x-axis exactly to the edges of the system
    ax.set_xlim(left=0, right=cuts[-1] + 1) 

    # 2. Labels and Grid
    ax.set_xlabel('Subsystem Size $|A|$', fontsize=18)
    
    # Dynamically extract L from the maximum cut size
    L_approx = int(cuts[-1] + cuts[0]) if len(cuts) > 1 else "L"
    ax.set_ylabel(f'Entanglement Entropy $S_A(|A|)$', fontsize=18) 
    ax.set_title(f'Steady-State Page Curve ($L={L_approx}$)', pad=15)
    
    ax.grid(True, linestyle=':', alpha=0.6, color='black')

    # 3. Legend Outside the Plot
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, 
              edgecolor='darkgrey', fontsize=14)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    print(f"-> Success! Triangle Page Curve safely generated at {output_file}")

if __name__ == '__main__':
    plot_full_page_curve()