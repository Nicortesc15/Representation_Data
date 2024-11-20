from scipy.linalg import svd
import utils

""" Task 1.2: In this script, we apply principal component analysis to a racoon image. 
We need functions defined in utils.py for this script.
"""

# TODO: Load and resize the racoon image in grayscale
img = utils.load_resize_image()
img_t = img.T                    # Columns considered as data points

# TODO: Compute Singular Value Decomposition (SVD) using utils.compute_svd()
centered_img_t = utils.center_data(img_t)
U, S, V_t = utils.compute_svd(centered_img_t)

# TODO: Reconstruct images using utils.reconstruct_images
utils.reconstruct_images(U,S,V_t)


# TODO: Compute the number of components where energy loss is smaller than 1% using utils.compute_num_components_capturing_threshold_energy()
utils.compute_num_components_capturing_threshold_energy(S)
