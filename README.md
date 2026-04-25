# Measurement-Induced Phase Transitions (MIPT) in Random Clifford Circuits


This repository contains code for the simulation of **Measurement-Induced Phase Transitions (MIPT)** in random Clifford circuits. This codebase simulates random circuits made up of 2-qubit Clifford gates and single-qubit Z-measurement operations to obtain entanglement entropy scaling properties, and is designed to reproduce the key numerical results from Li, Chen, and Fisher's paper on the "Quantum Zeno effect and the many-body entanglement transition" (see [References](#references)).

## Table of Contents
1. [Overview](#overview)
2. [Architecture & Directory Structure](#architecture--directory-structure)
3. [Key Methodologies](#key-methodologies)
4. [Installation & Requirements](#installation--requirements)
5. [Usage](#usage)
    - [1. Data Generation (Execution)](#1-data-generation-execution)
    - [2. Data Processing](#2-data-processing)
    - [3. Plotting & Analysis](#3-plotting--analysis)

---

## Overview

In dynamically evolving quantum systems, a phase transition occurs as the rate of local projective measurements $p$ increases. 
- At low $p$ (Volume-law phase), the system generates widespread global entanglement, and the entanglement entropy is proportional to system size.
- At high $p$ (Area-law phase), measurements project the wave-function, localizing information, and the entanglement entropy is proportional to the boundary of the subsystem. In 1D, this means the entanglement entropy is a constant, irrespective of system size.
- A critical point $p_c$ separates the two regimes.

This simulation pipeline constructs random discrete Clifford circuits and measures the scaling evolution of entanglement entropy. Utilizing the [`Stim`](https://github.com/quantumlib/Stim) quantum emulator, it rapidly constructs layered 2-qubit tableau networks. For Clifford circuits, we can keep track of the state by keeping track of the generators of the stabilizer group. For an N-qubit state, we record these generators as rows of an $N \times 2N$ binary matrix. The entanglement entropy of a subsystem is calculated by extracting the GF(2) matrix rank of the submatrix corresponding to the subsystem using bit-parallelized `Numba` functions. Since we use random circuits, we must take an ensemble average over many (~500) independent realizations of the random circuit to obtain statistically meaningful results. 

## Architecture & Directory Structure

```text
ph354-computational-physics-project/
├── data/                    # Simulated outputs mapped to .npz statistical aggregates
├── figures/                 # Exported plot and figures
└── src/                     
    ├── core/                # Lower-level physical formulations and algorithms
    │   ├── calculate_entropy.py # Numba-accelerated bit-packed GF(2) Rank Calculator
    │   ├── circuits.py          # Builds random 2-qubit Clifford circuits using Stim
    │   └── observables.py       # Extracts entanglement entropy as a function of different parameters
    │
    ├── execution/           # Orchestrated parallel executions for grid sweeps
    │   ├── dynamics.py          # Tracks entropy growth vs layers of circuit depth (time)
    │   ├── entropy_scaling.py   # Parameter sweep to extract half-chain entropy vs L, for different measurement rates
    │   └── page_curve.py        # Entropy scaling behavior measured at various sub-system sizes, for different measurement rates
    │
    ├── data_processing/     # Map-reduce routines to concatenate batch jobs
    │   ├── merge_page.py        # Compiles parallel Page curve splits
    │   └── merge_scaling_data.py# Concatenates simulation outputs for scaling data
    │
    └── plotting/            # Analysis and Publication Visualizations
        └── plot.py              # CLI utility wrapper for various plot formats
```

## Key Methodologies

### Stabilizer Simulation
To exceed the 25-30 qubit bottleneck in simulating quantum circuits by keeping track of the full density matrix, we restrict the gate pool exclusively to the **Clifford Group**. Using `Stim`, this limits operations purely to stabilizer state rotations, allowing us to represent our state by a binary check matrix (tableau). Stim allows us to track the tableau as we apply gates and measurements. Since we deal with large system sizes, we build build only one layer of the circuit at a time, discarding the previous layer. We precompile all 11520 2-qubit Clifford gates before starting the simulations to enable efficient sampling of the random Clifford gates. 


### Fast Gaussian Elimination Over Binary Matrices
One of the core computational bottlenecks of MIPT modeling is locating entanglement entropy (all Renyi entropies happen to be same for Clifford circuits!), given as:
$$S_A = \text{Rank}(A) - |A|$$
where $A$ is the GF(2) matrix of the stabilizer generators of the subsystem. In `src/core/calculate_entropy.py`, the system encodes boolean rows (stabilizer matrices) into 64-bit integer vectors (`uint64`). To obtain the rank of submatrix A, it performs a localized Gaussian elimination algorithm, processing 64 operations concurrently per clock cycle via `Numba` compilation, accelerating typical evaluation workflows drastically.

## Installation & Requirements

1. Make sure Python 3.9+ is installed.  
2. An active virtual environment (e.g. `stim_env`) is recommended.

*(Note: Numba is highly necessary for caching and compiling the C-bound bitwise array structures.)*

## Usage

### 1. Data Generation (Execution)
Simulation modules exist under `src/execution/` and implement multiprocessing by default to capitalize on multiple cores. Run these scripts with command line arguments to execute massive cluster tasks.

**Entropy Scaling:** sweeps $L$ (qubit size) vs. $p$ (projective rate)
```bash
python src/execution/entropy_scaling.py -L 32 64 128 -p 0.1 0.15 0.16 0.2 -N 1000
```
This saves an aggregated `.npz` to the `data/scaling/` directory encapsulating trajectories computed over $N=1000$ batches.

**Page Curve:** sweeps sub-system cuts.
```bash
python src/execution/page_curve.py -L 64 -p 0.10 0.16 0.20 -N 500
```

**Dynamics:** tracks entropy generation over circuit layers (time).
```bash
python src/execution/dynamics.py -L 32 64 -p 0.16 0.20 -N 500
```

### 2. Data Processing
For immense resolutions, simulations are often clustered. Rather than keeping them siloed, you can sew them laterally to consolidate the grids.
```bash
python src/data_processing/merge_scaling_data.py --input-dir data/scaling_raw --output-file data/scaling/master.npz
```

### 3. Plotting & Analysis
Generate robust, standardized PDFs using the unified entrypoint in `src/plotting/plot.py`.

```bash
# Generate half-chain Entropy vs p for specific lengths
python src/plotting/plot.py vs-p --file data/scaling/master.npz --L-values 32 64 128

# Generate a Log-Log scaling parameter mapping for Entropy vs L
python src/plotting/plot.py vs-L --file data/scaling/master.npz --p-values 0.1 0.16 0.2

# Build the Page Curve (Subsystem Size |A| vs Entropy)
python src/plotting/plot.py page --file data/page/master.npz 

# Extract Time scaling Dynamics over layers T
python src/plotting/plot.py dynamics --files data/dynamics/dynamics_L32-64_p0.16-0.20_N500.npz --target-p 0.16
```

## References

- Yaodong Li, Xiao Chen, and Matthew P. A. Fisher. "Quantum Zeno effect and the many-body entanglement transition." *Phys. Rev. B* **98**, 205136 (2018). DOI: [https://doi.org/10.1103/PhysRevB.98.205136](https://doi.org/10.1103/PhysRevB.98.205136)
