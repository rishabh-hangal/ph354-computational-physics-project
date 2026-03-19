import stim

import stim
print(stim.__version__)

circuit = stim.Circuit()
circuit.append("H", [0])
circuit.append("CX", [0, 1])

sim = stim.TableauSimulator()

# Apply the circuit to the simulator
sim.do_circuit(circuit)

tableau = sim.current_tableau()

print("Tableau:")
print(tableau)