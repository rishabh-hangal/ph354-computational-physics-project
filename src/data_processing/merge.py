import numpy as np
import os
import glob
import argparse
from src.config import SCALING_DATA_DIR, PAGE_DATA_DIR, ensure_dirs

def merge_page_curves(file1, file2, output_file):
    print(f"Loading {file1}...")
    d1 = np.load(file1)
    print(f"Loading {file2}...")
    d2 = np.load(file2)

    if not np.array_equal(d1['cuts'], d2['cuts']):
        print("Error: The 'cuts' arrays (L values) in both files do not match! Cannot merge.")
        return

    p_values = np.concatenate([d1['p_values'], d2['p_values']])
    cuts = d1['cuts']
    S_mean_page = np.vstack([d1['S_mean_page'], d2['S_mean_page']])
    
    # Try merging uncertainties and time if they exist
    S_var_page = None
    if 'S_var_page' in d1 and 'S_var_page' in d2:
        S_var_page = np.vstack([d1['S_var_page'], d2['S_var_page']])
        
    Time_per_p = None
    if 'Time_per_p' in d1 and 'Time_per_p' in d2:
        Time_per_p = np.concatenate([d1['Time_per_p'], d2['Time_per_p']])

    # Sort strictly by p_values
    sort_idx = np.argsort(p_values)
    p_values = p_values[sort_idx]
    S_mean_page = S_mean_page[sort_idx, :]
    
    if S_var_page is not None: S_var_page = S_var_page[sort_idx, :]
    if Time_per_p is not None: Time_per_p = Time_per_p[sort_idx]

    # Handle duplicates by taking the first overlapping instance (if p=0 was repeated)
    unique_ps, unique_idxs = np.unique(p_values, return_index=True)
    
    kwargs = {
        'p_values': unique_ps,
        'cuts': cuts,
        'S_mean_page': S_mean_page[unique_idxs, :]
    }
    
    if S_var_page is not None: kwargs['S_var_page'] = S_var_page[unique_idxs, :]
    if Time_per_p is not None: kwargs['Time_per_p'] = Time_per_p[unique_idxs]

    np.savez_compressed(output_file, **kwargs)
    print(f"Successfully merged {len(p_values)} -> {len(unique_ps)} unique p_values!")
    print(f"Saved into: {output_file}")


def merge_scaling_files(file_list, output_filename):
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
        p_vals = data['p_values']
        S_mean = data['S_mean']
        L_vals = data['L_values']
        
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

    combined_p = np.concatenate(all_p_values)
    combined_S = np.concatenate(all_S_means, axis=1)
    sort_indices = np.argsort(combined_p)
    sorted_p = combined_p[sort_indices]
    sorted_S = combined_S[:, sort_indices]
    
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

def main():
    parser = argparse.ArgumentParser(description="Unified plotting script for Data Processing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Page
    parser_page = subparsers.add_parser('page', help="Merge two page_curve npz files of same L.")
    parser_page.add_argument('--file1', required=True, help="First input file")
    parser_page.add_argument('--file2', required=True, help="Second input file")
    parser_page.add_argument('--out', required=True, help="Output file")

    # 2. Scaling
    parser_scaling = subparsers.add_parser('scaling', help="Merge scaling datasets.")
    parser_scaling.add_argument('--files', nargs='+', help="Specific files to merge (optional)")
    parser_scaling.add_argument('--out', default=None, help="Output file (optional)")

    args = parser.parse_args()

    ensure_dirs()

    if args.command == 'page':
        merge_page_curves(args.file1, args.file2, args.out)
    elif args.command == 'scaling':
        if args.files:
            files_to_merge = args.files
        else:
            # Defaults logic
            search_pattern = os.path.join(SCALING_DATA_DIR, "scaling_L*.npz")
            potential_files = glob.glob(search_pattern)
            files_to_merge = [f for f in potential_files if "master" not in f and "stitched" not in f and "ALL" not in f]
            
        out_file = args.out if args.out else os.path.join(SCALING_DATA_DIR, "scaling_master_stitched.npz")
        if len(files_to_merge) > 0:
            merge_scaling_files(files_to_merge, out_file)
        else:
            print("No valid scaling files found to merge.")

if __name__ == '__main__':
    main()
