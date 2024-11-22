import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.linalg import svd

""" Task 1.3: In this script, we apply principal component analysis to pedestrian trajectory data. 
We need functions defined in utils.py for this script.
"""

# TODO: Load trajectory data in data_DMAP_PCA_Vadere.txt. (Hint: You may need to use a space as delimiter)
data = np.loadtxt('data/data_DMAP_PCA_vadere.txt', dtype = np.float64, delimiter = " ")
# TODO: Center the data by subtracting the mean
data_centered = utils.center_data(data)
# TODO: Extract positions of pedestrians 1 and 2
p1 = data_centered[:,:2]
p2 = data_centered[:,2:4]
# TODO: Visualize trajectories of first two pedestrians (Hint: You can optionally use utils.visualize_traj_two_pedestrians() )
title = 'Trajectories of first two pedestrians'
xlabel = 'x position'
ylabel = 'y position'
legend = (title, xlabel, ylabel)
utils.visualize_traj_two_pedestrians(p1, p2, legend)
# TODO: Compute SVD of the data using utils.compute_svd()
U, S, V_t = utils.compute_svd(data_centered)
# TODO: Reconstruct data by truncating SVD using utils.reconstruct_data_using_truncated_svd()
recons_data = utils.reconstruct_data_using_truncated_svd(U, S, V_t, 2)
# TODO: Visualize trajectories of the first two pedestrians in the 2D space defined by the first two principal components
p1_recons = recons_data[:,:2]
p2_recons = recons_data[:,2:4]
utils.visualize_traj_two_pedestrians(p1_recons, p2_recons, legend)
# TODO: Answer the questionsin the worksheet with the help of utils.compute_cumulative_energy(), utils.compute_num_components_capturing_threshold_energy()



