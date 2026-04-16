import numpy as np
from numba import njit, test
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


### FUNCTION 1: RANK CALCULATION OF BINARY MATRICES USING BIT-PACKING AND NUMBA OPTIMIZATION
### ONE SINGLE FUNCTION FOR BOTH PACKING AND RANK CALCULATION, OPTIMIZED WITH NUMBA. THIS DOES NOT PRESERVE THE ORIGINAL MATRIX!
@njit
def rank_binary_matrix(binary_matrix):
    """
    Compresses an (N x M) binary matrix into an (N x W) uint64 matrix.
    Every 64 columns of the original matrix become 1 column of 64-bit integers.
    """
    rows, cols = binary_matrix.shape
    # Calculate how many 64-bit words we need per row (ceiling division)
    words = (cols + 63) // 64 
    
    # Initialize the compressed matrix with 64-bit unsigned integers
    packed = np.zeros((rows, words), dtype=np.uint64)
    
    """Iterate through the original binary matrix and set the corresponding bits in the packed matrix. We do this by 
    calculating the word index and bit index for each 1 in the original matrix, then using bitwise OR with a bit mask to set the bit in the packed matrix.
    The bitmask is created by left-shifting 1 by the bit index by an amount equal to the bit index, and we ensure that the shift is done on a 64-bit type to prevent overflow.
    The bitmask is then ORed with the current value in the packed matrix at the appropriate word index to set the bit corresponding to the 1 in the original matrix.
    We only process the 1s to save time, as the 0s are already"""
    for r in range(rows):
        for c in range(cols):
            if binary_matrix[r, c] == 1: 
                word_idx = c // 64 #extract the word index
                bit_idx = c % 64 #extract the bit index within that word

                # Force the 1 to be a 64-bit type before shifting to prevent overflow. 
                bit_mask = np.uint64(1) << np.uint64(bit_idx)
                packed[r, word_idx] |= bit_mask #manually set the bit using bitwise OR for each 1 in the original matrix
                
    """
    Computes the GF(2) rank directly on the packed uint64 memory blocks.
    Utilizes the upper-triangular shortcut to cut mathematical operations in half.
    """
    rows, words = packed.shape
    rank = 0
    
    for col in range(cols):
        # Locate the exact integer and the exact bit within it for the current column
        word_idx = col // 64
        bit_idx = col % 64
        
        # 1. Pivot search (Optimized: Only search from the current rank downwards)
        pivot_row = -1
        for r in range(rank, rows):
            # Shift the target bit to the 0th position and mask it with 1
            if (packed[r, word_idx] >> np.uint64(bit_idx)) & np.uint64(1):
                pivot_row = r
                break
                
        if pivot_row == -1:
            continue
            
        # 2. Row swap
        if pivot_row != rank:
            for w in range(words):
                temp = packed[rank, w]
                packed[rank, w] = packed[pivot_row, w]
                packed[pivot_row, w] = temp
                
        # 3. Row elimination via XOR 
        # (Optimized: Only eliminate strictly BELOW the pivot to form Row Echelon Form)
        for r in range(rank + 1, rows):
            if (packed[r, word_idx] >> np.uint64(bit_idx)) & np.uint64(1):
                # The Hardware Magic: XORing a single word processes 64 bits simultaneously
                for w in range(words):
                    packed[r, w] ^= packed[rank, w]
                    
        rank += 1
        if rank == rows:
            break
            
    return rank


### FUNCTION 2: ENTROPY CALCULATION FROM STABILIZER TABLEAU
def calculate_entropy(stabilizer, n_cut):
    """
    Calculates the entanglement entropy for a given stabilizer matrix and cut.
    """
    if n_cut > stabilizer.shape[0]:
        raise ValueError("n_cut cannot be greater than the number of qubits in the stabilizer.")
    L = stabilizer.shape[1] // 2
    logging.debug(f"Stabilizer shape: {stabilizer.shape}, L: {L}, n_cut: {n_cut}")  
    

    sub_A = np.hstack([stabilizer[:, :n_cut], stabilizer[:, L : L+n_cut]])
    logging.debug(f"Submatrix: \n {sub_A} \n")

    rank = rank_binary_matrix(sub_A)
    logging.debug(f"Rank: {rank}")

    return rank - n_cut