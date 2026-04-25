from setuptools import setup, find_packages

setup(
    name="ph354_mipt",
    version="0.1.0",
    description="Measurement-Induced Phase Transitions (MIPT) Simulation Pipeline",
    author="Rishabh Hangal",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "matplotlib",
        "pandas",
        "stim"
    ],
    python_requires=">=3.9",
)
