import numpy as np
import matplotlib.pyplot as plt
import utils

""" 
Task2.1: In this script, we demonstrate the similarity of Diffusion Maps and Fourier analysis using a periodic dataset.
We need functions defined in utils.py for this script.
"""

# Create a periodic dataset with the details described in the task-sheet
N = 1000  # number of points of the dataset
t_k = (2 * np.pi / (N + 1)) * np.arange(1, N + 1)
dataset = np.column_stack((np.cos(t_k), np.sin(t_k)))

# Visualize data-set
plt.plot(dataset[:, 0], dataset[:, 1])
plt.title("Periodic dataset")
plt.axis("scaled")
plt.show()

# Plot 5 eigenfunctions associated to the largest eigenvalues using the function diffusion_map() implemented in utils.py
n_eig = 5
lambda_, phi_l = utils.diffusion_map(dataset, n_eig)
for i in range(1, n_eig + 1):
    plt.plot(t_k, phi_l[:, i], label=f"$\phi_{i}$")
plt.title(f"First {n_eig} Eigenfunctions")
plt.xlabel(r"$t_k$")
plt.ylabel(r"$\phi$")
plt.legend()
plt.show()
