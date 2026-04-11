import stim
import numpy as np
import random
from entropy import calculate_entropy

def sim_to_matrix(sim):
    tableau = sim.current_inverse_tableau() ** -1
    _, _, z2x, z2z, _, _ = tableau.to_numpy()
    return np.hstack([z2x.astype(np.uint8), z2z.astype(np.uint8)])

def one_layer_circuit(L, p, parity, sim):
    """
    Generates a single layer of random 2-qubit Clifford gates followed by random measurements.
    - L: Total number of qubits (must be even)
    - p: Probability of measuring each qubit
    - parity: 0 for even pairs (0-1, 2-3, ...), 1 for odd pairs (1-2, 3-4, ...)
    This is useful for calculating entropy after each timestep.
    """
    if parity not in [0, 1]:
        raise ValueError("Parity must be 0 (even) or 1 (odd).")   
    for i in range(parity, L - 1, 2):
        sim.do_tableau(stim.Tableau.random(2), [i, i+1])
    measured = [q for q in range(L) if random.random() < p]
    sim.measure_many(*measured)

    return sim




def multilayer_circuit(L, T, p):
    """
    Generates a multi-layer circuit with random 2-qubit Clifford gates and measurements.
     Args:
    - L: Total number of qubits (must be even)
    - T: Number of layers
    - p: Probability of measuring each qubit

    Returns:
    - A binary stabilizer matrix representing the final state after T layers

    
    This function constructs the circuit layer by layer, applying the gates and measurements immediately after each layer.
    It does NOT build the entire circuit first and then execute it. The former saves memory and allows for intermediate state extraction if needed.
    """
    sim = stim.TableauSimulator()
    sim.set_num_qubits(L)

    for t in range(T):
        parity = t % 2
        one_layer_circuit(L, p, parity, sim)
    tableau = sim.current_inverse_tableau() ** -1 
    _, _, z2x, z2z, _, _ = tableau.to_numpy() 
    return np.hstack([z2x.astype(np.uint8), z2z.astype(np.uint8)])