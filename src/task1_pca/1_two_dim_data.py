import utils
import numpy as np
import matplotlib.pyplot as plt
""" Task 1.1: In this script, we apply principal component analysis to two-dimensional data. 
We need functions defined in utils.py for this script.
"""

# TODO: Load the dataset from the file pca_dataset.txt
data = np.loadtxt('data/pca_dataset.txt', dtype=float, delimiter=" ")

# TODO: Compute mean of the data
mean_data = np.mean(data, axis=0)

# TODO: Center data
data_centered = data - mean_data

# TODO: Compute SVD
U, S, V = np.linalg.svd(data_centered, full_matrices=True)
principal_directions = V.T

# TODO:Plot principal components

# Plot initial data centered 
plt.scatter(data_centered[:,0],data_centered[:,1], alpha=0.7, color = 'black', s =10)

# Calculate limits and plot directions of the principal components
limits = np.array([np.min(data_centered) - 0.5, np.max(data_centered) + 0.5])
colors = ['blue', 'orange']

for i in range(2):  
    direction = principal_directions[:, i]
    x = limits * direction[0]
    y = limits * direction[1]
    plt.plot(x, y, linestyle='--', color=colors[i])

plt.xlabel("x")
plt.ylabel("f(x)")
plt.axis("scaled")
plt.show()

# TODO: Analyze the energy captured by the first two principal components using utils.compute_energy()

