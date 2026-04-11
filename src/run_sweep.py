import numpy as np
import multiprocessing as mp
import time
import logging
import os

# Import the specific protocol you want to run from your observables module
from observables import measure_final_entropy

# ==========================================
# 1. LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# ==========================================
# 2. THE WORKER UNPACKER
# ==========================================
def worker_task(args):
    """
    Unpacks the parameters and runs exactly one quantum trajectory.
    """
    L, T, p = args
    return measure_final_entropy(L, T, p)

# ==========================================
# 3. THE MASTER ORCHESTRATOR
# ==========================================
def main():
    # --- Define the Parameter Grid ---
    L_values = [8, 16, 32, 64, 128, 256, 512, 1024]                                 # System sizes
    p_values = np.linspace(0.16, 0.20, 5)                   # Sweep probabilities
    num_shots = 500                                          # Trajectories per (L, p) point
    
    # Initialize matrices to store the statistical results
    S_mean = np.zeros((len(L_values), len(p_values)))
    S_variance = np.zeros((len(L_values), len(p_values)))
    
    # NEW: A 1D array to store the total time taken for ALL p's at a given L
    Time_per_L = np.zeros(len(L_values))
    
    total_simulations = len(L_values) * len(p_values) * num_shots
    logging.info(f"Initializing MIPT Sweep. Total physical trajectories: {total_simulations}")
    
    master_start_time = time.time()
    
    # --- The 2D Grid Sweep ---
    for i, L in enumerate(L_values):
        
        # Start the timer for this entire System Size (L)
        L_start_time = time.time()
        
        T = int(2 * L)
        
        for j, p in enumerate(p_values):
            
            tasks = [(L, T, p) for _ in range(num_shots)]
            
            with mp.Pool(processes=mp.cpu_count()) as pool:
                raw_entropies = pool.map(worker_task, tasks)
                
            S_mean[i, j] = np.mean(raw_entropies)
            S_variance[i, j] = np.var(raw_entropies)
            
            logging.info(f"Finished L={L:<4} | p={p:.4f} | Mean S={S_mean[i,j]:.4f}")
            
        # Stop the timer only after all p_values are done for this L
        L_total_time = time.time() - L_start_time
        Time_per_L[i] = L_total_time
        
        logging.info(f"*** COMPLETED ALL 'p' FOR L={L}. TOTAL TIME: {L_total_time:.2f}s ***\n")
            
    # --- Save the Data ---
    save_dir = "../data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "L=8-1024_p=0.16-0.20_N=500_mipt_finite_size_scaling.npz") #rename before each run to reflect the protocol and sweep details
    
    np.savez_compressed(
        save_path, 
        L_values=L_values, 
        p_values=p_values, 
        S_mean=S_mean, 
        S_variance=S_variance,
        Time_per_L=Time_per_L  # <--- Now saving the 1D array
    )
    
    total_time = time.time() - master_start_time
    logging.info(f"Sweep complete! Data safely written to {save_path}")
    logging.info(f"Total execution time: {total_time / 60:.2f} minutes")

# ==========================================
# 4. THE MULTIPROCESSING EXECUTION GUARD
# ==========================================
if __name__ == '__main__':
    main()