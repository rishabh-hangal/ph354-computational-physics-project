import numpy as np
from src.core.calculate_entropy import rank_binary_matrix, calculate_entropy

def test_rank_binary_matrix():
    # Example binary matrix
    M = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0]
    ], dtype=np.uint8)
    
    # rank of M should be 2 because row3 = row1 + row2 (mod 2)
    assert rank_binary_matrix(M) == 2

def test_calculate_entropy():
    # Dummy stabilizer matrix
    # Format (N, 2L). Let's say L=3 -> cols=6
    stabilizer = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
    ], dtype=np.uint8)
    
    # calculate_entropy(stabilizer, n_cut)
    ent = calculate_entropy(stabilizer, 1)
    assert isinstance(ent, int)
