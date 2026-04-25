import numpy as np
import multiprocessing as mp
import time
import logging
import os
import argparse
from src.config import DYNAMICS_DATA_DIR, ensure_dirs

# Import your dynamics function
from src.core.observables import entropy_over_time

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s: %(message)s', datefmt='%H:%M:%S')

def worker_task(args):
    """
    STRICT PARAMETER PASSING: 
    No default arguments are used here. Every physical variable required 
    by the simulation must be explicitly unpacked and passed.
    """
    L, T, p = args
    return entropy_over_time(L, T, p)

def main():
    parser = argparse.ArgumentParser(description="Sweep L and p values for entanglement dynamics.")
    parser.add_argument('-L', '--L_values', type=int, nargs='+', required=True, help="List of system sizes")
    parser.add_argument('-p', '--p_values', type=float, nargs='+', required=True, help="List of probabilities")
    parser.add_argument('-N', '--num_shots', type=int, required=True, help="Number of trajectories per point")
    args = parser.parse_args()

    # =========================================================================
    # CONFIGURATION ZONE
    # All physical and infrastructural parameters MUST be defined here.
    # Do not hardcode numbers anywhere else in the script.
    # =========================================================================
    
    L_values = np.array(args.L_values)
    p_values = np.array(args.p_values)
    
    num_shots = args.num_shots
    ensure_dirs()
    out_dir = DYNAMICS_DATA_DIR
    
    # Calculate uniform max time based on the LARGEST system size
    T_max = int(2 * np.max(L_values))
    
    # Generate an explicit filename based on the config
    L_str = f"L{np.min(L_values)}" if np.min(L_values) == np.max(L_values) else f"L{np.min(L_values)}-{np.max(L_values)}"
    p_str = f"p{np.min(p_values)}" if np.min(p_values) == np.max(p_values) else f"p{np.min(p_values)}-{np.max(p_values)}"
    
    filename = f"dynamics_{L_str}_{p_str}_N{num_shots}.npz"
    save_path = os.path.join(out_dir, filename)
    
    # =========================================================================
    
    logging.info("=========================================")
    logging.info("   INITIALIZING MASTER DYNAMICS SWEEP    ")
    logging.info("=========================================")
    logging.info(f"System sizes (L) : {L_values}")
    logging.info(f"Probabilities (p): {p_values}")
    logging.info(f"Uniform Time (T) : {T_max}")
    logging.info(f"Trajectories     : {num_shots}")
    logging.info(f"Target File      : {filename}")
    
    master_start = time.time()
    
    # 3D Tensors: (Number of Ls, Number of ps, Number of Timesteps)
    S_mean_master = np.zeros((len(L_values), len(p_values), T_max))
    S_var_master = np.zeros((len(L_values), len(p_values), T_max))
    
    # ==========================================
    # SIMULATION LOOPS
    # ==========================================
    for i, L in enumerate(L_values):
        logging.info(f"\n---> Starting sweep for L = {L} <---")
        
        for j, p in enumerate(p_values):
            p_start = time.time()
            
            # EXPLICIT PARAMETER PACKING
            tasks = [(L, T_max, p) for _ in range(num_shots)]
            
            with mp.Pool(processes=mp.cpu_count()) as pool:
                raw_results = pool.map(worker_task, tasks)
                
            results_matrix = np.vstack(raw_results)
            
            S_mean_master[i, j, :] = np.mean(results_matrix, axis=0)
            S_var_master[i, j, :] = np.var(results_matrix, axis=0)
            
            step_time = time.time() - p_start
            logging.info(f"  [L={L}] Finished p={p:.3f} | Final S={S_mean_master[i, j, -1]:.4f} | Time: {step_time:.2f}s")

    # ==========================================
    # DATA ARCHIVING
    # ==========================================
    # The output file is fully self-describing. It contains the data 
    # AND the exact parameters used to generate the data.
    np.savez_compressed(
        save_path, 
        
        # Metadata / Parameters
        L_values=L_values, 
        p_values=p_values, 
        time_steps=np.arange(T_max),
        num_shots=num_shots,
        T_max=T_max,
        
        # Data
        S_mean_master=S_mean_master, 
        S_var_master=S_var_master
    )
    
    logging.info("\n=========================================")
    logging.info(f"ALL SIMULATIONS COMPLETE IN {(time.time() - master_start) / 60:.2f} MINS")
    logging.info(f"Self-describing archive saved to: {save_path}")

if __name__ == '__main__':
    main()