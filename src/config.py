import os

# Base directory: the root of the project (parent of src)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SCALING_DATA_DIR = os.path.join(DATA_DIR, 'scaling')
PAGE_DATA_DIR = os.path.join(DATA_DIR, 'page')
DYNAMICS_DATA_DIR = os.path.join(DATA_DIR, 'dynamics')

# Figures directory
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

def ensure_dirs():
    """Ensure that the necessary output directories exist."""
    for d in [DATA_DIR, SCALING_DATA_DIR, PAGE_DATA_DIR, DYNAMICS_DATA_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
