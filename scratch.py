import stim
import numpy as np
# Number of random Clifford gates
num_gates = 11520 # Size of 2-qubit Clifford group
# Precompile all 11520 2-qubit Clifford gates. Actually, it's easier to just sample random tableaus and keep a cache of some size, or just pregenerate a large number of random tableaus if the set of 11520 is hard to iterate.
# Wait, let's look at `stim.Tableau.random(2)`
# The user wants to "precompile all Clifford gates".
# The 2-qubit Clifford group has 11520 elements. Iterating over all 11520 exact elements is not natively supported by stim (unless we iterate over 11520 integers and use `stim.Tableau.from_conjugated_generators`).
# Maybe the user just means "pre-generate an array of random Clifford gates" rather than exactly "all" unique ones, or maybe just cache a large number.
# Wait, Stim has `stim.Tableau.iter_all(2)`? Let's check.
