import stim
import numpy as np

sim = stim.TableauSimulator()
sim.h(0)

tableau = sim.current_inverse_tableau().inverse()
xs2xs, xs2zs, zs2xs, zs2zs = tableau.to_numpy()

check_matrix = np.block([[xs2xs.astype(int), xs2zs.astype(int)],
                          [zs2xs.astype(int), zs2zs.astype(int)]])
print(check_matrix)
