import utils
import numpy as np
import matplotlib.pyplot as plt
""" Task 1.1: In this script, we apply principal component analysis to two-dimensional data. 
We need functions defined in utils.py for this script.
"""

# TODO: Load the dataset from the file pca_dataset.txt
data = np.loadtxt('data/pca_dataset.txt', dtype=np.float64, delimiter=" ")

# TODO: Compute mean of the data
data_mean = np.mean(data, axis=0)

# TODO: Center data
data_centered = utils.center_data(data) 

# TODO: Compute SVD
U, S, V_t = utils.compute_svd(data_centered)
principal_components = V_t.T
print(S.shape)

# TODO:Plot principal components

# Plot initial data centered 
plt.scatter(data_centered[:,0],data_centered[:,1], alpha=0.7, color = 'black', s =10)

# Calculate limits and plot directions of the principal components
limits = np.array([np.min(data_centered) - 0.5, np.max(data_centered) + 0.5])
colors = ['blue', 'orange']

for i, color in enumerate(colors):
    direction = principal_components[:, i]
    x, y = limits * direction[0], limits * direction[1]
    plt.plot(x, y, linestyle='--', color=color)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.axis("scaled")
plt.show()

# TODO: Analyze the energy captured by the first two principal components using utils.compute_energy()

energy_pc1 = utils.compute_energy(S, 1)
energy_pc2 = utils.compute_energy(S, 2)

# Pie chart of energy percentage
labels = ['PC1', 'PC2']
pcts = [energy_pc1, energy_pc2]  # Percentages
colors = ['skyblue', 'orange']
plt.pie(pcts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Energy Distribution')
plt.show()

# Reconstrunction of data on a reduced dimensionality r
r = 1
X_r = utils.reconstruct_data_using_truncated_svd(U,S,V_t,r)

# Plot of the reconstructed centered data
plt.scatter(X_r[:,0],X_r[:,1], alpha=0.7, color = 'black', s =10)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()