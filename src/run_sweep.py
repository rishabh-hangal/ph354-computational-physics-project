import numpy as np
import multiprocessing as mp
import time
import logging
import os

# Import the specific protocol you want to run from your observables module
# Assuming your file is named observables.py and contains measure_final_entropy
from observables import measure_final_entropy

# ==========================================
# 1. LOGGING CONFIGURATION
# ==========================================
# Level INFO suppresses debug spam but shows high-level progress.
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
    The multiprocessing map function strictly requires the input to be a single 
    object. We pass a tuple of arguments and unpack them here for the worker core.
    """
    L, T, p = args
    # This core simulates exactly one universe and returns the scalar entropy
    return measure_final_entropy(L, T, p)

# ==========================================
# 3. THE MASTER ORCHESTRATOR
# ==========================================
def main():
    # --- Define the Parameter Grid ---
    L_values = [16, 32, 64]                                # System sizes
    p_values = np.linspace(0, 0.25, 26)                   # Sweep probabilities
    num_shots = 100                                          # Trajectories per (L, p) point
    
    # Initialize matrices to store the statistical results
    S_mean = np.zeros((len(L_values), len(p_values)))
    S_variance = np.zeros((len(L_values), len(p_values)))
    
    total_simulations = len(L_values) * len(p_values) * num_shots
    logging.info(f"Initializing MIPT Sweep. Total physical trajectories: {total_simulations}")
    
    master_start_time = time.time()
    
    # --- The 2D Grid Sweep ---
    for i, L in enumerate(L_values):
        
        # In random brickwork circuits, saturation typically occurs at T = 2L
        T = int(1.75 * L)
        
        for j, p in enumerate(p_values):
            step_start = time.time()
            
            # 1. Create the task list. 
            # We bundle the arguments into a tuple exactly 'num_shots' times.
            tasks = [(L, T, p) for _ in range(num_shots)]
            
            # 2. Spawn the worker pool.
            # mp.cpu_count() automatically detects all physical/logical cores on your chip.
            with mp.Pool(processes=mp.cpu_count()) as pool:
                # pool.map distributes the 500 tasks across the cores and blocks 
                # until every single core has reported back with its final float value.
                raw_entropies = pool.map(worker_task, tasks)
            
            # 3. Calculate the ensemble statistics
            S_mean[i, j] = np.mean(raw_entropies)
            S_variance[i, j] = np.var(raw_entropies)
            
            step_time = time.time() - step_start
            logging.info(f"Finished L={L:<4} | p={p:.4f} | Mean S={S_mean[i,j]:.4f} | Time: {step_time:.2f}s")
            
    # --- Save the Data ---
    # Ensure the target directory exists
    save_dir = "../data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "mipt_finite_size_scaling.npz")
    
    # np.savez_compressed writes multiple arrays into a single highly compressed binary file
    np.savez_compressed(
        save_path, 
        L_values=L_values, 
        p_values=p_values, 
        S_mean=S_mean, 
        S_variance=S_variance
    )
    
    total_time = time.time() - master_start_time
    logging.info(f"Sweep complete! Data safely written to {save_path}")
    logging.info(f"Total execution time: {total_time / 60:.2f} minutes")

# ==========================================
# 4. THE MULTIPROCESSING EXECUTION GUARD
# ==========================================
if __name__ == '__main__':
    main()