import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import argparse

# PUBLICATION FORMATTING
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
    'lines.markersize': 5
})

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'figures'))
UNITARY_OFFSET = 0.44
p_c = 0.16

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def scalar_formatter(val, pos):
    if val >= 1: return f"{int(val)}"
    return f"{val:g}"

def plot_vs_L(data_file, p_values_target=None):
    print(f"Loading data from {data_file}...")
    data = np.load(data_file)
    L_values = data['L_values']
    p_values = data['p_values']
    S_mean = data['S_mean']

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    markers = ['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h', '8']
    
    if p_values_target is None:
        p_values_target = p_values

    colors = plt.cm.tab20(np.linspace(0, 1, max(10, len(p_values_target))))
    
    for i, target_p in enumerate(p_values_target):
        p_idx = (np.abs(p_values - target_p)).argmin()
        actual_p = p_values[p_idx]
        S_line = S_mean[:, p_idx] + UNITARY_OFFSET
        safe_S = np.where(S_line > 0, S_line, np.nan)
        marker_style = markers[i % len(markers)]
        
        if abs(actual_p - p_c) < 0.005:
            ax.plot(L_values, safe_S, marker=marker_style, label=f'$p = {actual_p:.2f}$', 
                    color='black', linewidth=3.5, markerfacecolor='none', zorder=10)
        else:
            ax.plot(L_values, safe_S, marker=marker_style, label=f'$p = {actual_p:.2f}$', 
                    color=colors[i], markerfacecolor='none')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(scalar_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(scalar_formatter))

    ax.grid(False) 
    ax.set_xlim(left=L_values[0]*0.8, right=L_values[-1]*1.2)
    ax.set_xlabel('$L$', fontsize=18)
    ax.set_ylabel('$S_A(p; |A| = L/2, L)$', fontsize=18) 
    ax.legend(loc='upper left', frameon=True, edgecolor='darkgrey', fontsize=12, handlelength=1.5, ncol=2)
    
    ensure_dir(FIGURES_DIR)
    output_file = os.path.join(FIGURES_DIR, "S_vs_L.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"-> Saved {output_file}")

def plot_page_curve(data_file, p_values_target=None):
    print(f"Loading data from {data_file}...")
    data = np.load(data_file)
    p_values = data['p_values']
    cuts = data['cuts']            
    S_mean = data['S_mean_page']   
    
    L_actual = int(np.max(cuts) + 1)
    half_chain_mask = cuts <= (L_actual // 2)
    cuts_half = cuts[half_chain_mask]
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    if p_values_target is None:
        p_values_target = p_values

    colors = plt.cm.tab20(np.linspace(0, 1, max(10, len(p_values_target))))
    markers = ['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h', '8']
    
    for i, target_p in enumerate(p_values_target):
        p_idx = (np.abs(p_values - target_p)).argmin()
        actual_p = p_values[p_idx]
        
        S_line_half = S_mean[p_idx, half_chain_mask] + UNITARY_OFFSET
        cuts_plot = cuts_half[::2]
        S_plot = S_line_half[::2]
        
        safe_S = np.where(S_plot > 0, S_plot, np.nan)
        marker_style = markers[i % len(markers)]
        
        if abs(actual_p - 0.16) < 0.005:
            ax.plot(cuts_plot, safe_S, marker=marker_style, label=f'$p = {actual_p:.2f}$', 
                    color='black', linewidth=3.5, markerfacecolor='none', zorder=10)
        else:
            ax.plot(cuts_plot, safe_S, marker=marker_style, label=f'$p = {actual_p:.2f}$', 
                    color=colors[i], markerfacecolor='none')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(scalar_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(scalar_formatter))

    ax.grid(False) 
    ax.set_xlim(left=max(0.8, cuts_half[0]*0.8), right=cuts_half[-1]*1.1)
    ax.set_xlabel('$|A|$', fontsize=18)
    ax.set_ylabel(f'$S_A(p; |A|, L={L_actual})$', fontsize=18) 
    ax.legend(loc='upper left', frameon=True, edgecolor='darkgrey', fontsize=12, handlelength=1.5, ncol=2)
    
    ensure_dir(FIGURES_DIR)
    output_file = os.path.join(FIGURES_DIR, "page_curve.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"-> Saved {output_file}")

def plot_vs_p(data_file, L_values_target=None):
    print(f"Loading data from {data_file}...")
    data = np.load(data_file)
    L_values = data['L_values']
    p_values = data['p_values']
    S_mean = data['S_mean']

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p']
    
    if L_values_target is None:
        L_values_target = L_values

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(L_values_target))))

    for i, target_L in enumerate(L_values_target):
        if target_L not in L_values:
            print(f"Warning: L={target_L} not found in data.")
            continue
        L_idx = np.where(L_values == target_L)[0][0]
        
        # Apply the analytical offset
        S_line = S_mean[L_idx, :] + UNITARY_OFFSET
        
        marker_style = markers[i % len(markers)]
        ax.plot(p_values, S_line, marker=marker_style, label=f'$L={target_L}$', 
                color=colors[i], markerfacecolor='none')

    ax.set_yscale('log')
    # Use a baseline of 0.2 to comfortably frame the 0.44 physical bottom 
    # without cropping anything, maintaining the view all the way to p=1.
    ax.set_ylim(bottom=0.2, top=1000)
    ax.set_xlim(left=-0.02, right=1.02) 
    ax.yaxis.set_major_formatter(FuncFormatter(scalar_formatter))

    ax.grid(False) 
    ax.xaxis.grid(True, linestyle=':', alpha=0.6, color='black')

    ax.set_xlabel('$p$', fontsize=18)
    ax.set_ylabel('$S_A(L/2)$', fontsize=18) 
    ax.legend(loc='upper right', frameon=True, edgecolor='darkgrey', fontsize=14, handlelength=1.5)
    
    ensure_dir(FIGURES_DIR)
    output_file = os.path.join(FIGURES_DIR, "S_vs_p.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"-> Saved {output_file}")

def plot_dynamics(files, target_p):
    print(f"Plotting Dynamics from {len(files)} file(s) for p={target_p}...")
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(files)*3)) # Enough colors
    actual_p = None
    
    color_idx = 0
    for f in files:
        if not os.path.exists(f):
            print(f"Missing {f}")
            continue
        print(f"Processing {f}...")
        data = np.load(f)
        p_values = data['p_values']
        time_steps = data['time_steps']

        # Determine index of p
        p_idx = (np.abs(p_values - target_p)).argmin()
        actual_p = p_values[p_idx]
        
        # Check structure: earlier files were 3D or 2D depending on single L vs multiple L
        if 'S_mean_master' in data: # Shape (L, p, t)
            L_values = data['L_values']
            for i, L in enumerate(L_values):
                entropy_vs_time = data['S_mean_master'][i, p_idx, :] + UNITARY_OFFSET
                time_plot = time_steps[::2] if len(time_steps) > len(entropy_vs_time) else time_steps
                ax.plot(time_plot, entropy_vs_time[::2] if len(entropy_vs_time) > len(time_plot) else entropy_vs_time, 
                        label=f'L = {L}', color=colors[color_idx])
                color_idx += 1
        else: # e.g. mipt_dynamics_L64.npz, Shape (p, t)
            # Try to infer L from filename or data
            if 'L' in data:
                L = data['L']
            else:
                try:
                    L = int(f.split('L')[-1].split('.')[0].split('_')[0])
                except:
                    L = "Unknown"
            
            entropy_vs_time = data['S_mean_time'][p_idx, :] + UNITARY_OFFSET
            
            # Simple length matching
            if len(entropy_vs_time) == len(time_steps):
                t_plot, e_plot = time_steps, entropy_vs_time
            elif len(entropy_vs_time) * 2 == len(time_steps):
                t_plot, e_plot = time_steps[::2], entropy_vs_time
            else:
                t_plot, e_plot = time_steps, entropy_vs_time
                diff = len(t_plot) - len(e_plot)
                if diff > 0: t_plot = t_plot[:-diff]
                elif diff < 0: e_plot = e_plot[:diff]

            ax.plot(t_plot, e_plot, label=f'L = {L}', color=colors[color_idx])
            color_idx += 1

    if actual_p is None:
        print("No valid files processed.")
        return

    ax.set_title(f'Entanglement Growth Scaling ($p = {actual_p:.3f}$)', pad=15)
    ax.set_xlabel('Time ($t$)', fontsize=16)
    ax.set_ylabel('Entanglement Entropy $S(t)$', fontsize=16)
    ax.legend(loc='lower right', frameon=False)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)
    
    ensure_dir(FIGURES_DIR)
    output_file = os.path.join(FIGURES_DIR, f"dynamics_scaling_p{actual_p:.2f}.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"-> Saved {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Unified plotting script for MIPT data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. vs-L
    parser_vsl = subparsers.add_parser('vs-L', help="Plot Entropy vs System Size L")
    parser_vsl.add_argument('--file', required=True, help="Path to scaling data .npz file")
    parser_vsl.add_argument('--p-values', type=float, nargs='+', help="Specific p values to plot")

    # 2. page
    parser_page = subparsers.add_parser('page', help="Plot Log-Log Half-Chain Page Curve")
    parser_page.add_argument('--file', required=True, help="Path to page curve .npz file")
    parser_page.add_argument('--p-values', type=float, nargs='+', help="Specific p values to plot")

    # 3. vs-p
    parser_vsp = subparsers.add_parser('vs-p', help="Plot Entropy vs Probability p")
    parser_vsp.add_argument('--file', required=True, help="Path to scaling data .npz file")
    parser_vsp.add_argument('--L-values', type=int, nargs='+', help="Specific L values to plot")

    # 4. dynamics
    parser_dyn = subparsers.add_parser('dynamics', help="Plot Entropy vs Time")
    parser_dyn.add_argument('--files', required=True, nargs='+', help="Path to one or more dynamics .npz files")
    parser_dyn.add_argument('--target-p', required=True, type=float, help="Target p value to extract across files")

    args = parser.parse_args()

    if args.command == 'vs-L':
        plot_vs_L(args.file, args.p_values)
    elif args.command == 'page':
        plot_page_curve(args.file, args.p_values)
    elif args.command == 'vs-p':
        plot_vs_p(args.file, args.L_values)
    elif args.command == 'dynamics':
        plot_dynamics(args.files, args.target_p)

if __name__ == '__main__':
    main()
