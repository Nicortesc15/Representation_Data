import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.linalg import svd

""" Task 1.3: In this script, we apply principal component analysis to pedestrian trajectory data. 
We need functions defined in utils.py for this script.
"""

# TODO: Load trajectory data in data_DMAP_PCA_Vadere.txt. (Hint: You may need to use a space as delimiter)

# TODO: Center the data by subtracting the mean

# TODO: Extract positions of pedestrians 1 and 2

# TODO: Visualize trajectories of first two pedestrians (Hint: You can optionally use utils.visualize_traj_two_pedestrians() )

# TODO: Compute SVD of the data using utils.compute_svd()

# TODO: Reconstruct data by truncating SVD using utils.reconstruct_data_using_truncated_svd()

# TODO: Visualize trajectories of the first two pedestrians in the 2D space defined by the first two principal components

# TODO: Answer the questionsin the worksheet with the help of utils.compute_cumulative_energy(), utils.compute_num_components_capturing_threshold_energy()



