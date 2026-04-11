import numpy as np
import matplotlib.pyplot as plt
import os

def plot_finite_size_scaling(file_path, num_shots):
    """
    Loads the MIPT data and plots the final entanglement entropy 
    across different system sizes to identify the critical point p_c.
    """
    # 1. LOAD THE DATA
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}. Did the sweep finish?")
        
    data = np.load(file_path)
    
    # Extract the arrays using their dictionary keys
    L_values = data['L_values']
    p_values = data['p_values']
    S_mean = data['S_mean']
    S_variance = data['S_variance']
    
    # 2. CALCULATE THE STANDARD ERROR OF THE MEAN (SEM)
    # Applying the equation: SEM = sqrt(Variance / N)
    S_error = np.sqrt(S_variance / num_shots)
    
    # 3. CONFIGURE MATPLOTLIB FOR PUBLICATION QUALITY
    plt.figure(figsize=(8, 6), dpi=150)
    plt.rcParams.update({'font.size': 12})
    
    # Choose a clean color map for the different system sizes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 4. PLOT THE CURVES
    for i, L in enumerate(L_values):
        color = colors[i % len(colors)]
        
        # Plot the main ensemble average line
        plt.plot(p_values, S_mean[i], marker='o', markersize=4, 
                 linestyle='-', linewidth=1.5, color=color, 
                 label=f'L = {L}')
        
        # Plot the shaded error region (Mean + SEM, Mean - SEM)
        plt.fill_between(p_values, 
                         S_mean[i] - S_error[i], 
                         S_mean[i] + S_error[i], 
                         color=color, alpha=0.2)
                         
    # 5. FORMAT THE GRAPH
    plt.title('Measurement-Induced Phase Transition', pad=15)
    plt.xlabel('Measurement Probability ($p$)', fontsize=14)
    plt.ylabel('Half-Chain Entanglement Entropy $\\langle S(L/2) \\rangle$', fontsize=14)
    
    # Set the y-axis to a logarithmic scale
    plt.yscale('log')
    
    # Add a vertical grid for easier reading of the crossing point
    # We can also add a horizontal grid for the minor log ticks to make it easier to read
    plt.grid(True, which='major', linestyle='--', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    
    plt.legend(title="System Size", loc='best')
    
    # Add a vertical grid for easier reading of the crossing point
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="System Size", loc='best')
    
    # Save the figure to your PDF folder, then display it
    save_path = "../figures/trial_run_finite_size_scaling_raw.pdf"
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    plt.show()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == '__main__':
    # Define the exact number of shots you used in run_sweep.py
    N_SHOTS = 100 
    DATA_FILE = "../data/trial_run_mipt_finite_size_scaling.npz"
    
    plot_finite_size_scaling(DATA_FILE, N_SHOTS)

def plot_data_collapse(file_path, p_c_guess, nu_guess):
    """
    Applies the finite-size scaling ansatz x = (p - p_c) * L^(1/nu) 
    to collapse the entropy curves onto a single universal function.
    """
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}")
        
    data = np.load(file_path)
    L_values = data['L_values']
    p_values = data['p_values']
    S_mean = data['S_mean']
    
    plt.figure(figsize=(8, 6), dpi=150)
    plt.rcParams.update({'font.size': 12})
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, L in enumerate(L_values):
        color = colors[i % len(colors)]
        
        # 1. THE MATH: Rescale the x-axis using the derived ansatz
        x_scaled = (p_values - p_c_guess) * (L ** (1.0 / nu_guess))
        
        # 2. THE MATH: Shift the y-axis so all curves meet at the origin
        # We find the entropy value closest to p_c for this specific L
        idx_c = np.argmin(np.abs(p_values - p_c_guess))
        S_critical = S_mean[i, idx_c]
        y_shifted = S_mean[i] - S_critical
        
        # Plot the collapsed data
        plt.plot(x_scaled, y_shifted, marker='o', markersize=4, 
                 linestyle='-', linewidth=1.5, color=color, 
                 label=f'L = {L}')
                 
    # Format the collapsed graph
    plt.title(f'Data Collapse ($p_c={p_c_guess}$, $\\nu={nu_guess}$)', pad=15)
    
    # Using LaTeX string formatting for the x-axis label
    plt.xlabel(r'Scaling Variable $x = (p - p_c)L^{1/\nu}$', fontsize=14)
    plt.ylabel(r'$S(p, L) - S(p_c, L)$', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1) # Mark the critical point
    plt.legend(title="System Size", loc='best')
    
    save_path = "../figures/mipt_data_collapse.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Collapse plot saved to {save_path}")
    
    plt.show()

# To use this, add it to the bottom of your analyze.py execution block:
if __name__ == '__main__':
    DATA_FILE = "../data/trial_run_mipt_finite_size_scaling.npz"
    
    # Standard 1D random Clifford MIPT values are roughly:
    # p_c ~ 0.16, \nu ~ 1.3. You will need to tune these to fit your specific data!
    plot_data_collapse(DATA_FILE, p_c_guess=0.16, nu_guess=1.33)

    import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def optimize_data_collapse(L_values, p_values, S_mean, p_c_bounds, nu_bounds):
    """
    Finds the critical parameters p_c and nu that minimize the scatter 
    of the finite-size scaling data collapse.
    """
    
    def cost_function(params):
        p_c_guess, nu_guess = params
        
        # Penalize the optimizer heavily if it wanders outside physical bounds
        if not (p_c_bounds[0] < p_c_guess < p_c_bounds[1]) or not (nu_bounds[0] < nu_guess < nu_bounds[1]):
            return 1e6 
            
        x_pooled = []
        y_pooled = []
        
        for i, L in enumerate(L_values):
            # 1. Interpolate to find S(p_c, L) for the y-shift
            # We use cubic interpolation for smooth, physically accurate estimates
            interpolator = interp1d(p_values, S_mean[i], kind='cubic', fill_value="extrapolate")
            S_critical = interpolator(p_c_guess)
            
            # 2. Rescale the axes
            x_scaled = (p_values - p_c_guess) * (L ** (1.0 / nu_guess))
            y_shifted = S_mean[i] - S_critical
            
            x_pooled.extend(x_scaled)
            y_pooled.extend(y_shifted)
            
        x_pooled = np.array(x_pooled)
        y_pooled = np.array(y_pooled)
        
        # 3. Fit a 3rd-degree polynomial "master curve" to the pooled data
        # np.polyfit returns the coefficients, np.polyval evaluates them
        poly_coeffs = np.polyfit(x_pooled, y_pooled, deg=3)
        y_fit = np.polyval(poly_coeffs, x_pooled)
        
        # 4. Calculate the Mean Squared Error (The Cost)
        mse = np.mean((y_pooled - y_fit) ** 2)
        return mse

    # Set the initial guesses (the starting coordinate for the optimizer)
    initial_guess = [np.mean(p_c_bounds), np.mean(nu_bounds)]
    
    # Run the Nelder-Mead optimization algorithm
    print("Starting optimization... Please wait.")
    result = minimize(
        cost_function, 
        initial_guess, 
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6}
    )
    
    if result.success:
        best_pc, best_nu = result.x
        print(f"\nOptimization Successful!")
        print(f"Optimal p_c = {best_pc:.5f}")
        print(f"Optimal nu  = {best_nu:.5f}")
        print(f"Final Minimum Cost (MSE) = {result.fun:.2e}")
        return best_pc, best_nu
    else:
        raise RuntimeError("Optimization failed to converge. Check your bounds or data quality.")
    
def plot_computational_scaling(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}")
        
    data = np.load(file_path)
    L_values = data['L_values']
    p_values = data['p_values']
    
    # Load the 1D array of total execution times
    Time_per_L = data['Time_per_L']
    num_p = len(p_values)
    
    # --- PLOTTING ---
    plt.figure(figsize=(8, 6), dpi=150)
    plt.rcParams.update({'font.size': 12})
    
    plt.plot(L_values, Time_per_L, marker='s', markersize=8, 
             linestyle='-', linewidth=2.5, color='#2ca02c')
                 
    # Set to Log-Log Scale to reveal algorithmic complexity
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    
    plt.title(f'Algorithmic Scaling (Total time for all {num_p} probabilities)', pad=15)
    plt.xlabel('System Size ($L$)', fontsize=14)
    plt.ylabel('Total Execution Time (Seconds)', fontsize=14)
    
    plt.xticks(L_values, labels=[str(L) for L in L_values])
    plt.grid(True, which='major', linestyle='-', alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    
    save_path = "../figures/computational_scaling.pdf"
    os.makedirs("../figures", exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Scaling plot saved to {save_path}")
    
    plt.show()

# To run this, just add it to your execution block:


# ==========================================
# HOW TO INTEGRATE THIS INTO YOUR SCRIPT
# ==========================================
if __name__ == '__main__':
    data = np.load("../data/trial_run_mipt_finite_size_scaling.npz")
    plot_computational_scaling("../data/trial_run_mipt_finite_size_scaling.npz")
    # We restrict the optimizer to a physically reasonable window 
    # based on the MIPT universality class literature.
    optimal_pc, optimal_nu = optimize_data_collapse(
        L_values=data['L_values'],
        p_values=data['p_values'],
        S_mean=data['S_mean'],
        p_c_bounds=(0.10, 0.25),
        nu_bounds=(0.8, 2.0)
    )
    
    # Now you plug these optimal values directly into the plotting function 
    # we wrote previously!
    # plot_data_collapse("../data/trial_run_mipt_finite_size_scaling.npz", optimal_pc, optimal_nu)