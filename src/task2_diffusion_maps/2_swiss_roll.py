import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import utils

""" 
Task2.2: In this script, we compute eigenfunctions of the Laplace Beltrami operator on the
“swiss roll” manifold.  We need functions defined in utils.py for this script.
"""

# Generate swiss roll dataset
data_points = 5000
swiss_roll_data, t = make_swiss_roll(data_points, noise=0.0)

# Visualize data-set
elevation = 10  # Vertical angle (degrees)
azimuth = 78    # Horizontal angle (degrees)

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection="3d")
ax.view_init(elev=elevation, azim=azimuth)
scatter = ax.scatter(swiss_roll_data[:, 0], swiss_roll_data[:, 1], swiss_roll_data[:, 2], c=t ,s=10, alpha=0.8)
ax.set_title("Swiss Roll")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# Use function diffusion_map() defined in utils to compute first ten eigenfunctions (corresponding to 10 largest eigenvalues) 
# of the Laplace Beltrami operator on the “swiss roll” manifold
n_eig = 12
lambda_, phi_l = utils.diffusion_map(swiss_roll_data, n_eig)

# Plot of first non-constant eigenfunction φ1 against the other eigenfunctions
for i in range(2,n_eig):
    plt.figure()
    plt.scatter(phi_l[:, 1], phi_l[:, i], c=t, s=1, alpha=0.5)
    plt.title(f'$\phi_1$ vs $\phi_{{{i}}}$')
    plt.xlabel(r'$\phi_1$')
    plt.ylabel(f'$\phi_{{{i}}}$')
    plt.show()

# PCA computation
U, S, V_t = utils.compute_svd(swiss_roll_data)  # Decompose swiss roll into singular vectors U, V and values S
n_components = [2, 3]                            # Number of principal components

# Plots with n principal components
for n in n_components:
    data_reconstructed = utils.reconstruct_data_using_truncated_svd(U, S, V_t,n)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    ax.scatter(data_reconstructed[:, 0], data_reconstructed[:, 1], data_reconstructed[:, 2], c=t , s=10, alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f'Reconstruction Using {n} First Principal Components')
    plt.show()

