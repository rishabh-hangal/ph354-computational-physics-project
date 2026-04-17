import numpy as np
import multiprocessing as mp
import time
import logging
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your Page curve function
from core.observables import measure_page_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s: %(message)s', datefmt='%H:%M:%S')

def worker_task(args):
    """Unpacks args and returns a 1D array of entropy vs spatial cut."""
    L, T, p = args
    return measure_page_curve(L, T, p)
def main():
    parser = argparse.ArgumentParser(description="Sweep p values for Page curve.")
    parser.add_argument('-L', '--L', type=int, required=True, help="System size")
    parser.add_argument('-p', '--p_values', type=float, nargs='+', required=True, help="List of probabilities")
    parser.add_argument('-N', '--num_shots', type=int, required=True, help="Number of trajectories per point")
    args = parser.parse_args()

    L = args.L                                     # Fixed system size
    T = int(2 * L)                              # Evolve to steady state
    p_values = np.array(args.p_values)      # Sweep probabilities across the transition
    num_shots = args.num_shots
    
    # Initialize matrix: rows = different p values, columns = spatial cuts (1 to L-1)
    S_mean_page = np.zeros((len(p_values), L - 1))
    S_var_page = np.zeros((len(p_values), L - 1))
    
    Time_per_p = np.zeros(len(p_values))
    
    logging.info(f"Initializing Page Curve Sweep for L={L}. Total trajectories: {len(p_values) * num_shots}")
    master_start = time.time()
    
    for j, p in enumerate(p_values):
        p_start = time.time()
        
        tasks = [(L, T, p) for _ in range(num_shots)]
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            raw_results = pool.map(worker_task, tasks)
            
        # Stack the list of 1D arrays into a 2D matrix (num_shots x L-1)
        results_matrix = np.vstack(raw_results)
        
        # Average across the shots to get the final Page curve for this p
        S_mean_page[j, :] = np.mean(results_matrix, axis=0)
        S_var_page[j, :] = np.var(results_matrix, axis=0)
        
        step_time = time.time() - p_start
        Time_per_p[j] = step_time
        
        # The half-chain entropy is at the exact middle of the array
        mid_idx = (L - 1) // 2
        logging.info(f"Finished p={p:.3f} | Half-chain S={S_mean_page[j, mid_idx]:.4f} | Time: {step_time:.2f}s")
            
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'page_curve'))
    os.makedirs(save_dir, exist_ok=True)
    
    p_str = f"p{np.min(p_values)}" if np.min(p_values) == np.max(p_values) else f"p{np.min(p_values)}-{np.max(p_values)}"
    save_path = os.path.join(save_dir, f"page_curve_L{L}_{p_str}_N{num_shots}.npz")
    
    np.savez_compressed(save_path, p_values=p_values, cuts=np.arange(1, L), 
                        S_mean_page=S_mean_page, S_var_page=S_var_page, Time_per_p=Time_per_p)
    
    logging.info(f"Page Curve Sweep complete! Saved to {save_path} in {(time.time() - master_start) / 60:.2f} mins")

if __name__ == '__main__':
    main()