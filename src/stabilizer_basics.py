import stim
import numpy as np

# 1. Create the 'Instruction' set
circuit = stim.Circuit()
circuit.append("S", [0])  
circuit.append("H", [0])
circuit.append("S", [1])
circuit.append("H", [1])

# 2. Create the 'Simulator' (The thing that holds the state)
sim = stim.TableauSimulator()

# 3. Feed the instructions to the simulator
sim.do(circuit)

# 4. Peek at the Binary Check Matrix (Tableau)
# We use current_inverse_tableau because it gives the 'stabilizers' directly
# Pull the internal Heisenberg state and invert it to get the Schrödinger state
tableau = sim.current_inverse_tableau() ** -1
print(f"Tableau for 1 qubit:\n{tableau}")

# 1. to_numpy() returns 6 blocks: 
# (X-to-X, X-to-Z, Z-to-X, Z-to-Z, x_signs, z_signs)
x2x, x2z, z2x, z2z, x_signs, z_signs = tableau.to_numpy()

# 2. Extract the Stabilizers (Z-generators). 
# We horizontally stack the X-components and Z-components.
stabilizer_matrix = np.hstack([z2x, z2z])

# Convert boolean (True/False) to integers (1/0) for your GF(2) solver
stabilizer_matrix = stabilizer_matrix.astype(int)

print(f"\nRaw Binary Check Matrix (L x 2L):\n{stabilizer_matrix}")