import scipy as sp
from scipy.linalg import eigh
from scipy.sparse import lil_matrix
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt

""" This script contains the implementation of the diffusion map algorithm. 
"""

def create_distance_matrix(X, max_distance=200):
    """ Compute a sparse distance matrix using scipy.spatial.KDTree. Set max_distance as 200.

    Args:
        X (npt.NDArray[np.float]): Data matrix.
        max_distance (int, optional): Computes a distance matrix leaving as zero any distance greater than max_distance. Defaults to 200.

    Returns:
        npt.NDArray[np.float]: Distance Matrix (Hint: You may have to use (D).toarray()!, output shape = (np.shape(D)[0], np.shape(D)[0]))
    """
    # TODO: Implement method
    # Hints: using scipy.spatial.KDTree, set max_distance as 200, you may have to use .toarray() to the array you are returning!)
    N = X.shape[0]                                                  # Number of points in the data-set
    sparse_matrix = lil_matrix((N,N))                               # Initialize sparse matrix

    tree = KDTree(X)                                                # Create a tree to look up the nearest neighbors of a point

    for i in range(N):
        radius_points = tree.query_ball_point(X[i], max_distance)   # Row index of the points inside the radius
        for j in radius_points:
            if i != j:
                distace = np.linalg.norm(X[i] - X[j])               # Distance between neighboring points
                sparse_matrix[i, j] = distace
    
    return sparse_matrix.toarray()

def set_epsilon(p, distance_matrix):
    """ Set scalar epsilon as 'p' % of the diameter of the dataset.
    (Step 2 of the algorithm mentioned in the worksheet.)

    Args:
        p (np.float64): percentage.
        distance_matrix (npt.NDArray[np.float]): Distance matrix.

    Returns:
        np.float64: returns epsilon.
    """
    # TODO: Implement method (Hint: p is a float between 1-100, you have to divide by 100)
    max_distance = distance_matrix.max()
    eps =  p * max_distance / 100 

    return eps

def create_kernel_matrix(D, eps):
    """Create the Kernel matrix.

    Args:
        D (npt.NDArray[np.float]): Distance matrix
        eps (np.float64): epsilon.

    Returns:
        npt.NDArray[np.float]: Kernel matrix. (output shape = (np.shape(D)[0], np.shape(D)[0]))
    """
    # TODO: Form the kernel matrix W (Step 3 of the algorithm from the worksheet)
    W = np.exp((-D**2 / eps))

    # TODO: Form the diagonal normalization matrix (Step 4 of the algorithm from the worksheet)
    P = np.diag(np.sum(W, axis = 1))
    # TODO: Normalize W to form the kernel matrix K (Step 5 of the algorithm from the worksheet)
    P_inv = np.diag(1 / np.diag(P))
    K = P_inv @ W @ P_inv

    return K


def diffusion_map(X, n_eig_vals=5):
    """ Implementation of the diffusion map algorithm.
        Please refer to the algorithm in the worksheet for the following.
        The step numbers in the following refer to the steps of the algorithm in the worksheet.

    Args:
        X (npt.NDArray[np.float]): Data matrix (each row represents one data point)
        n_eig_vals (int, optional): The number of eigenvalues and eigenvectors of the Laplace-Beltrami operator defined on the manifold close to the data to be computed. Default is 10.

    Returns:
        tuple(npt.NDArray[np.float], npt.NDArray[np.float]): eigenvalues, eigenvector of the Laplace-Beltrami operator
        output shapes: (n_eig_vals + 1, ), (np.shape(X)[0], n_eig_vals + 1)
    """

    # TODO: Compute distance matrix. Use method create_distance_matrix(..) defined in this script. (Step 1 from the algorithm in the worksheet)
    D = create_distance_matrix(X)
    # TODO: Use function set_epsion(.., ..) defined in this script to set epsilon to 5% of the diameter of the dataset (Step 1 from the algorithm in the worksheet).
    eps = set_epsilon(5, D)
    # TODO: Form Kernel matrix K. Use function create_kernel_matrix(.., ..) defined in this script. (Steps 3-5 from the algorithm in the worksheet) 
    K = create_kernel_matrix(D, eps)
    # TODO: Form the diagonal normalization matrix (Step 6 from the algorithm in the worksheet)
    Q_diag = np.sum(K, axis = 1)
    Q = np.diag(Q_diag)
    # TODO: Form symmetric matrix T_hat (Step 7 from the algorithm in the worksheet)
    Q_inv_sqrt = np.diag(1 / np.sqrt(Q_diag))
    T_hat = Q_inv_sqrt @ K @ Q_inv_sqrt
    # TODO: Find the L + 1 largest eigenvalues and the corresponding eigenvectors of T_hat (Step 8 from the algorithm in the worksheet)
    eig_vals, eig_vects = eigh(T_hat)
    a_l = np.flip(eig_vals)[:(n_eig_vals + 1)]
    v_l = np.flip(eig_vects, axis = 1)[:, :(n_eig_vals + 1)]
    # TODO: Compute the eigenvalues of T_hat^(1/ε) in DESCENDING ORDER (Hint: You can use np.flip(..))!! (Step 9 from the algorithm in the worksheet)
    lambda_ = np.sqrt(a_l**(1 / eps))
    # TODO: Compute the eigenvectors of the matrix T (Hint: You can use np.flip(..) with an appropriate axis) (Step 10 from the algorithm in the worksheet)
    phi_l = Q_inv_sqrt @ v_l
    
    return lambda_, phi_l

def center_data(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ Center data by subtracting the mean of the data. 

    Args:
        data (npt.NDArray[np.float64]): Data matrix.

    Returns:
        npt.NDArray[np.float64]: centered data.
    """
    # TODO: Implement method 
    data_mean = np.mean(data, axis=0)
    data_centered = data - data_mean
    return data_centered

def compute_svd(data: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """ Compute (reduced) SVD of the data matrix. Set (full_matrices=False).
    
    Args:
        data (npt.NDArray[np.float]): data matrix.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: U, S, V_t.
    """

    # TODO: Implement method
    U, S, Vt = np.linalg.svd(data, full_matrices=False)
    return U, S, Vt

def reconstruct_data_using_truncated_svd(U:npt.NDArray[np.float64], S:npt.NDArray[np.float64], V_t:npt.NDArray[np.float64], n_components:int):
    """ This function takes in the SVD of the data matrix and reconstructs the data matrix by retaining only 'n_components' SVD components.
    In other words, it computes a low-rank approximation with (rank = n_components) of the data matrix. 

    Args:
        U (npt.NDArray[np.float64]): Matrix whose columns contain left singular vectors
        S (npt.NDArray[np.float64]): Matrix with singular values
        V_t (npt.NDArray[np.float64]): Matrix whose rows contain right singular vectors
        n_components (int): no. of principal components retained in the low-rank approximation

    Returns:
        npt.NDArray[np.float64]: Reconstructed matrix using first 'n_components' principal components.
    """
    # TODO: Implement method
    U_r = U[:, :n_components]               # Obtain the r columns of U corresponding to largest singular values
    S_r = np.diag(S[:n_components])         # Create a diagonal matrix with the first r singular values from S
    Vt_r = V_t[:n_components, :]            # Obtain the first r rows of Vt (eigendirections)
    X_r = U_r @ S_r @ Vt_r                  # Reconstruction of the data 
    return X_r