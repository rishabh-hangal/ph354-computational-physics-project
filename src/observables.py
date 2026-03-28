import stim
import numpy as np
from entropy import calculate_entropy
from circuits import sim_to_matrix, one_layer_circuit, multilayer_circuit, 


def entropy_over_time(L, T, p):
    """
    Simulates a random circuit with measurements and calculates the entanglement entropy after each layer.
    Returns a list of entropies at each time step.
    """
    sim = stim.TableauSimulator()

    entropies = []
    
    for t in range(T):
        parity = t % 2
        one_layer_circuit(L, p, parity, sim)
        stabilizer_matrix = sim_to_matrix(sim)
        entropy = calculate_entropy(stabilizer_matrix, n_cut=L//2)
        entropies.append(entropy)
        
    return entropies


def measure_final_entropy(L, T, p):
    final_matrix = multilayer_circuit(L, T, p)
    return calculate_entropy(final_matrix, n_cut=L//2)

def measure_page_curve(L, T, p):
    final_matrix = multilayer_circuit(L, T, p)
    entropies = []
    for n_cut in range(1, L):
        entropy = calculate_entropy(final_matrix, n_cut=n_cut)
        entropies.append(entropy)
    return entropies