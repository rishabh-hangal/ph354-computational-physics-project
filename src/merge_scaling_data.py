import numpy as np
import glob
import os

def merge_scaling_files(file_list, output_filename):
    """
    Loads multiple fragmented MIPT scaling .npz files, concatenates their 
    p_values and S_mean arrays, sorts them by p, and saves a master file.
    """
    print(f"Attempting to merge {len(file_list)} files...")
    
    all_p_values = []
    all_S_means = []
    master_L_values = None
    
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"  [!] Warning: {file_path} not found. Skipping.")
            continue
            
        print(f"  -> Loading {file_path}")
        data = np.load(file_path)
        
        # Extract arrays
        p_vals = data['p_values']
        S_mean = data['S_mean']
        L_vals = data['L_values']
        
        # Ensure all files share the exact same system sizes (L)
        if master_L_values is None:
            master_L_values = L_vals
        else:
            if not np.array_equal(master_L_values, L_vals):
                raise ValueError(f"CRITICAL ERROR: L_values mismatch in {file_path}. Cannot merge.")
        
        all_p_values.append(p_vals)
        all_S_means.append(S_mean)
        
    if not all_p_values:
        print("No valid files were loaded. Exiting.")
        return

    # 1. Concatenate the lists of arrays into single massive arrays
    # p_values is a 1D array: [p1, p2, p3, ...]
    combined_p = np.concatenate(all_p_values)
    
    # S_mean is a 2D array of shape (num_L, num_p). 
    # We must concatenate strictly along the p-axis (axis=1).
    combined_S = np.concatenate(all_S_means, axis=1)
    
    # 2. Sort the arrays to prevent zig-zag lines in the plot
    # np.argsort returns the indices that would sort the combined_p array
    sort_indices = np.argsort(combined_p)
    
    # Apply these exact indices to both arrays to keep the data locked together
    sorted_p = combined_p[sort_indices]
    sorted_S = combined_S[:, sort_indices]
    
    # 3. Save the unified, perfectly sorted master file
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    np.savez_compressed(
        output_filename,
        L_values=master_L_values,
        p_values=sorted_p,
        S_mean=sorted_S
    )
    
    print("\n=========================================")
    print(f"SUCCESS: Merged {len(combined_p)} total data points.")
    print(f"Master file saved to: {output_filename}")
    print("=========================================")

if __name__ == '__main__':
    # Define the exact files you want to stitch together
    # Modify these paths to match your actual filenames
    files_to_merge = [
        "../data/L=8-1024_p=0.00-0.15_N=500_mipt_finite_size_scaling.npz",
        "../data/L=8-1024_p=0.16-0.20_N=500_mipt_finite_size_scaling.npz",
        "../data/L=8-1024_p=0.21_N=500_mipt_finite_size_scaling.npz",
        "../data/L=8-1024_p=0.22_N=500_mipt_finite_size_scaling.npz",
        "../data/L=8-1024_p=0.23_N=500_mipt_finite_size_scaling.npz",
        "../data/L=8-1024_p=0.24_N=500_mipt_finite_size_scaling.npz",
        "../data/L=8-1024_p=0.25-0.70_N=500_mipt_finite_size_scaling.npz"
    ]
    
    output_file = "../data/mipt_scaling_master_stitched.npz"
    
    merge_scaling_files(files_to_merge, output_file)