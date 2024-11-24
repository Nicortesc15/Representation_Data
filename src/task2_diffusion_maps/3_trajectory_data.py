import numpy as np
import matplotlib.pyplot as plt
import utils

"""Task2.3: In this script, we demonstrate the similarity of Diffusion Maps and Fourier analysis using a periodic dataset.
We need functions defined in utils.py for this script.
"""

# TODO: Create a periodic dataset with the details described in the task-sheet
data = np.loadtxt('data/data_DMAP_PCA_vadere.txt', dtype = np.float64, delimiter = " ")

# TODO: Obtain eigenvalues and eigenfunction using diffusion map
n_eig = 5
lambda_, phi_l = utils.diffusion_map(data, n_eig)

# TODO: Plot the first non-constant eigenfunction φ1 against the other eigenfunctions
# Plot of first non-constant eigenfunction φ1 against φ0
plt.figure()
plt.scatter(phi_l[:, 1], phi_l[:, 0], s=1, alpha=0.5)
plt.title(f'phi_1 vs phi_0')
plt.xlabel('phi_1')
plt.ylabel(f'phi_0')
plt.ylim((-0.1,0.1))
plt.show()

# Plot of first non-constant eigenfunction φ1 against the other eigenfunctions
for i in range(2,n_eig):
    plt.figure()
    plt.scatter(phi_l[:, 1], phi_l[:, i], s=1, alpha=0.5)
    plt.title(f'phi_1 vs phi_{i}')
    plt.xlabel('phi_1')
    plt.ylabel(f'phi_{i}')
    plt.axis('scaled')
    plt.show()




