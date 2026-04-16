import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# ==========================================
# PUBLICATION FORMATTING
# ==========================================
plt.rcParams.update({
    'font.size': 14,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'lines.linewidth': 2.5,
    'lines.markersize': 6
})

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ==========================================
# 1. PLOT: ENTANGLEMENT DYNAMICS
# ==========================================
def plot_dynamics(file_path):
    print("Plotting Dynamics...")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = np.load(file_path)
    p_values = data['p_values']
    time_steps = data['time_steps']
    S_mean = data['S_mean_time']
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    indices_to_plot = [0, len(p_values)//4, len(p_values)//2, -1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for c, idx in zip(colors, indices_to_plot):
        p = p_values[idx]
        entropy_vs_time = S_mean[idx, :]
        ax.plot(time_steps, entropy_vs_time, label=f'$p = {p:.3f}$', color=c)
        
    ax.set_title('Entanglement Dynamics (Half-Chain)', pad=15)
    ax.set_xlabel('Time ($t$)', fontsize=16)
    ax.set_ylabel('Entanglement Entropy $S(t)$', fontsize=16)
    ax.legend(loc='lower right', frameon=False)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ensure_dir("../figures")
    plt.savefig("../figures/mipt_dynamics.pdf", bbox_inches='tight')
    plt.close()
    print("-> Saved to figures/mipt_dynamics.pdf")

# ==========================================
# 2. PLOT: PAGE CURVE
# ==========================================
def plot_page_curve(file_path):
    print("Plotting Page Curve...")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = np.load(file_path)
    p_values = data['p_values']
    cuts = data['cuts']
    S_mean = data['S_mean_page']
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    indices_to_plot = [0, len(p_values)//4, len(p_values)//2, -1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for c, idx in zip(colors, indices_to_plot):
        p = p_values[idx]
        entropy_vs_cut = S_mean[idx, :]
        ax.plot(cuts, entropy_vs_cut, marker='o', label=f'$p = {p:.3f}$', color=c)
        
    ax.set_title('Steady-State Page Curve', pad=15)
    ax.set_xlabel('Subsystem Size ($x$)', fontsize=16)
    ax.set_ylabel('Entanglement Entropy $S(x)$', fontsize=16)
    ax.legend(loc='upper right', frameon=False)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ensure_dir("../figures")
    plt.savefig("../figures/mipt_page_curve.pdf", bbox_inches='tight')
    plt.close()
    print("-> Saved to figures/mipt_page_curve.pdf")

# ==========================================
# 3. PLOT: FINITE-SIZE SCALING
# ==========================================
def plot_finite_size_scaling(file_path):
    print("Plotting Finite-Size Scaling...")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = np.load(file_path)
    L_values = data['L_values']
    p_values = data['p_values']
    S_mean = data['S_mean'] 
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_values)))
    
    for i, L in enumerate(L_values):
        entropy_vs_p = S_mean[i, :]
        ax.plot(p_values, entropy_vs_p, marker='s', label=f'$L = {L}$', color=colors[i])
        
    ax.set_title('Finite-Size Scaling across the Transition', pad=15)
    ax.set_xlabel('Measurement Probability ($p$)', fontsize=16)
    ax.set_ylabel('Steady-State Half-Chain Entropy', fontsize=16)
    ax.legend(loc='upper right', frameon=False)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.axvline(x=0.16, color='black', linestyle=':', alpha=0.5, label='Approx $p_c \sim 0.16$')
    
    ensure_dir("../figures")
    plt.savefig("../figures/mipt_finite_size_scaling.pdf", bbox_inches='tight')
    plt.close()
    print("-> Saved to figures/mipt_finite_size_scaling.pdf")


# ==========================================
# 4. PLOT: DYNAMICS SCALING (Varying L)
# ==========================================
def plot_dynamics_scaling(data_dir, target_p):
    """
    Plots S(t) vs t for different system sizes L at a fixed measurement rate p.
    """
    print(f"Plotting Dynamics Scaling for p ≈ {target_p}...")
    
    L_values = [16, 32, 64]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_values)))
    
    actual_p = None
    
    for i, L in enumerate(L_values):
        file_path = os.path.join(data_dir, f"mipt_dynamics_L{L}.npz")
        
        if not os.path.exists(file_path):
            print(f"  -> Missing {file_path}, skipping L={L}.")
            continue
            
        data = np.load(file_path)
        p_values = data['p_values']
        time_steps = data['time_steps']
        S_mean = data['S_mean_time']
        
        # Find the index of the p_value closest to our target_p
        p_idx = (np.abs(p_values - target_p)).argmin()
        actual_p = p_values[p_idx]
        
        entropy_vs_time = S_mean[p_idx, :]
        
        ax.plot(time_steps, entropy_vs_time, label=f'$L = {L}$', color=colors[i])
        
    if actual_p is None:
        print("No data found to plot.")
        return

    ax.set_title(f'Entanglement Growth Scaling ($p = {actual_p:.3f}$)', pad=15)
    ax.set_xlabel('Time ($t$)', fontsize=16)
    ax.set_ylabel('Entanglement Entropy $S(t)$', fontsize=16)
    ax.legend(loc='lower right', frameon=False)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ensure_dir("../figures")
    save_path = f"../figures/mipt_dynamics_scaling_p{actual_p:.2f}.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"-> Saved to {save_path}")
# ==========================================
# MASTER EXECUTION (CLI)
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate specific MIPT plots.")
    parser.add_argument('--dynamics', action='store_true', help="Plot S vs t for varying p")
    parser.add_argument('--page', action='store_true', help="Plot S vs x for varying p")
    parser.add_argument('--scaling', action='store_true', help="Plot S vs p for varying L")
    
    # NEW: Now accepts a float value directly from the terminal
    parser.add_argument('--dyn-scaling', type=float, metavar='P', 
                        help="Plot S vs t for varying L at a specific measurement rate p")
    
    parser.add_argument('--all', action='store_true', help="Generate all plots (uses p=0.16 for dyn-scaling)")
    
    args = parser.parse_args()
    
    dynamics_file = "../data/mipt_dynamics_L64.npz"
    page_curve_file = "../data/p=0.02,0.25_mipt_page_curve_L128.npz"
    scaling_file = "../data/trial_run_mipt_finite_size_scaling.npz"
    data_dir = "../data"
    
    # Check if NO arguments were provided
    if not any(vars(args).values()) and args.dyn_scaling is None:
        print("Please specify which plot to generate.")
        print("Available flags: --dynamics, --page, --scaling, --dyn-scaling [P], --all")
        print("Example: python src/plot_results.py --dyn-scaling 0.16")
    else:
        if args.dynamics or args.all:
            plot_dynamics(dynamics_file)
            
        if args.page or args.all:
            plot_page_curve(page_curve_file)
            
        if args.scaling or args.all:
            plot_finite_size_scaling(scaling_file)
            
        # NEW: Pass the user's custom 'p' into the function
        if args.dyn_scaling is not None:
            plot_dynamics_scaling(data_dir, target_p=args.dyn_scaling)
        elif args.all:
            # Fallback for '--all' if a specific p wasn't chosen
            plot_dynamics_scaling(data_dir, target_p=0.16)