import numpy as np
import matplotlib.pyplot as plt
import os
import os
from src.config import FIGURES_DIR, SCALING_DATA_DIR, ensure_dirs

# Publication formatting
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

def scalar_formatter(val, pos):
    if val >= 1: return f"{int(val)}"
    return f"{val:g}"

def calc_residual(params, p_vals, L_vals, S_mean):
    pc, nu = params
    if not (0.01 < pc < 0.99) or not (0.05 < nu < 15):
        return 1e9
        
    xs_list = []
    ys_list = []
    
    for i, L in enumerate(L_vals):
        S_L = S_mean[i, :]
        EE_pc = np.interp(pc, p_vals, S_L)
        x = (p_vals - pc) * (L ** (1.0 / nu))
        y = S_L - EE_pc
        xs_list.append(x)
        ys_list.append(y)
        
    xs = np.concatenate(xs_list)
    ys = np.concatenate(ys_list)
    
    sort_idx = np.argsort(xs)
    xs = xs[sort_idx]
    ys = ys[sort_idx]
    
    tol = (xs[-1] - xs[0]) / 100.0 if xs[-1] > xs[0] else 0.01
    R = 0.0
    i = 0
    n = len(xs)
    while i < n:
        j = int(np.searchsorted(xs, xs[i] + tol, side='left'))
        j = max(j, i + 1)
        if j - i > 1:
            bucket = ys[i:j]
            R += np.var(bucket) * len(bucket)
        i = j
    return R

def custom_nelder_mead(f, x_start, args=(), step=0.1, max_iter=200):
    dim = len(x_start)
    simplex = np.zeros((dim + 1, dim))
    simplex[0] = x_start
    for i in range(dim):
        pt = np.array(x_start, copy=True)
        pt[i] += step
        simplex[i + 1] = pt
        
    f_vals = np.array([f(p, *args) for p in simplex])
    
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(max_iter):
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]
        
        best, f_best = simplex[0], f_vals[0]
        worst, f_worst = simplex[-1], f_vals[-1]
        second_worst = f_vals[-2]
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + alpha * (centroid - worst)
        fr = f(xr, *args)
        
        if f_best <= fr < second_worst:
            simplex[-1], f_vals[-1] = xr, fr
            continue
            
        if fr < f_best:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe, *args)
            if fe < fr:
                simplex[-1], f_vals[-1] = xe, fe
            else:
                simplex[-1], f_vals[-1] = xr, fr
            continue
            
        if fr >= second_worst:
            if fr < f_worst:
                xc = centroid + rho * (xr - centroid)
                fc = f(xc, *args)
                if fc <= fr:
                    simplex[-1], f_vals[-1] = xc, fc
                    continue
            else:
                xc = centroid - rho * (centroid - worst)
                fc = f(xc, *args)
                if fc < f_worst:
                    simplex[-1], f_vals[-1] = xc, fc
                    continue
                    
        for i in range(1, len(simplex)):
            simplex[i] = best + sigma * (simplex[i] - best)
            f_vals[i] = f(simplex[i], *args)
            
    order = np.argsort(f_vals)
    return simplex[order][0], f_vals[order][0]

def optimize_collapse(p_vals, L_vals, S_mean):
    print("Optimising scaling parameters with custom Nelder-Mead...")
    best_res = np.inf
    best_params = [0.16, 1.3]
    for pc_start in [0.10, 0.16, 0.20]:
        for nu_start in [0.8, 1.3, 1.8]:
            params, res = custom_nelder_mead(calc_residual, [pc_start, nu_start], args=(p_vals, L_vals, S_mean), step=0.03, max_iter=150)
            if res < best_res:
                best_res = res
                best_params = params
    return best_params[0], best_params[1], best_res

def plot_collapse(data_file, p_values_target=None):
    print(f"Loading data from {data_file}...")
    data = np.load(data_file)
    L_values = data['L_values']
    p_values_raw = data['p_values']
    S_mean_raw = data['S_mean']

    # Filter critical window
    mask = (p_values_raw >= 0.119) & (p_values_raw <= 0.201) & (~np.isclose(p_values_raw, 0.16))
    p_values = p_values_raw[mask]
    S_mean = S_mean_raw[:, mask]

    p_c, nu, R = optimize_collapse(p_values, L_values, S_mean)
    print(f"Optimised p_c = {p_c:.5f}, nu = {nu:.5f} (Residual = {R:.5e})")

    p_c_idx = np.where(np.isclose(p_values_raw, p_c))[0]
    if len(p_c_idx) == 0:
        S_pc = []
        for i in range(len(L_values)):
            S_pc.append(np.interp(p_c, p_values_raw, S_mean_raw[i, :]))
        S_pc = np.array(S_pc)
    else:
        S_pc = S_mean_raw[:, p_c_idx[0]]

    if p_values_target is not None:
        valid_indices = []
        for p_target in p_values_target:
             idx = np.abs(p_values - p_target).argmin()
             valid_indices.append(idx)
        valid_indices = sorted(list(set(valid_indices)))
    else:
        valid_indices = np.arange(len(p_values))

    p_plot = p_values[valid_indices]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    # Matching the paper's L values (approx)
    target_L = [32, 64, 128, 256, 512, 1024]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_L)))

    for i, L in enumerate(target_L):
        if L not in L_values:
            continue
        L_idx = np.where(L_values == L)[0][0]
        
        # x = (p - p_c) * L^(1/nu)
        x = (p_plot - p_c) * (L ** (1.0 / nu))
        
        # y = |S(p, L) - S(p_c, L)|
        y = np.abs(S_mean[L_idx, valid_indices] - S_pc[L_idx])
        
        # Exclude exactly p_c, and any 0y values for log scale
        valid_idx = (~np.isclose(p_plot, p_c)) & (y > 0)
        x_plot = x[valid_idx]
        y_plot = y[valid_idx]

        ax.plot(x_plot, y_plot, 'o', label=f'$L={L}$', color=colors[i], markersize=5, fillstyle='full', markeredgecolor='none')

    ax.set_yscale('log')
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(scalar_formatter))

    ax.axvline(0, color='black', linestyle='--', linewidth=1.0)
    
    yticks = [0.5, 1, 5, 10, 50, 100]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(t) for t in yticks])

    ax.set_xlabel(r'$(p - p_c) \ L^{1/\nu}$', fontsize=18)
    ax.set_ylabel(r'$|S_A(p, L/2) - S_A(p_c, L/2)|$', fontsize=16)

    ax.legend(loc='upper right', frameon=True, edgecolor='darkgrey', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6, color='black')

    ax.set_xlim(-14, 17.5)
    ax.set_ylim(0.2, 150)

    ensure_dirs()
    out_path = os.path.join(FIGURES_DIR, "scaling_collapse_fig6b.pdf")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Plot data collapse")
    parser.add_argument('--p-values', type=float, nargs='+', help="Specific p values to plot")
    args = parser.parse_args()

    data_path = os.path.join(SCALING_DATA_DIR, 'scaling_master_stitched.npz')
    plot_collapse(data_path, p_values_target=args.p_values)
