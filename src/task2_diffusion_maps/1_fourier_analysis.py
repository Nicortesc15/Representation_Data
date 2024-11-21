import numpy as np
import matplotlib.pyplot as plt
import utils

""" Task2.1: In this script, we demonstrate the similarity of Diffusion Maps and Fourier analysis using a periodic dataset.
We need functions defined in utils.py for this script.
"""

# TODO: Create a periodic dataset with the details described in the task-sheet
N = 1000  #number of points of the dataset
t_k = (2 * np.pi / (N + 1)) * np.arange(1, N + 1)
dataset = np.column_stack((np.cos(t_k), np.sin(t_k)))
print (dataset)
# TODO: Visualize data-set
plt.plot(dataset[:,0], dataset[:,1])
plt.title('Periodic dataset')
plt.show()
# TODO: Plot 5 eigenfunctions associated to the largest eigenvalues using the function diffusion_map() implemented in utils.py

# TODO: Plot 5 eigenfunctions



