import numpy as np
import matplotlib.pyplot as plt
import utils

"""Task2.3: In this script, we demonstrate the similarity of Diffusion Maps and Fourier analysis using a periodic dataset.
We need functions defined in utils.py for this script.
"""

# TODO: Create a periodic dataset with the details described in the task-sheet
data = np.loadtxt('data/data_DMAP_PCA_vadere.txt', dtype = np.float64, delimiter = " ")
# TODO: Visualize data-set

# TODO: Compute eigenfunctions associated to the largest eigenvalues using function diffusion_map() implemented in utils.py

# TODO: Plot plot the first non-constant eigenfunction φ1 against the other eigenfunctions


