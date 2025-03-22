import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

from shape_creator import *
from laplacian_creator import *

def eigenvalues_and_eigenvectors(laplacian, method, k=22):
    """
    Calculate the eigenvalues and eigenvectors of a Laplacian matrix.

    Parameters:
        laplacian (ndarray): The Laplacian matrix.
        method (str): The method to use for the eigenvalue calculation.
        k (int): The number of eigenvalues to calculate.

    Returns:
        tuple: (eigenvalues, eigenvectors)
            - eigenvalues (ndarray): The eigenvalues of the Laplacian matrix.
            - eigenvectors (ndarray): The eigenvectors of the Laplacian matrix.
    """
    method = method.lower()
    if method == 'eig':
        eigenvalues, eigenvectors = scipy.linalg.eig(laplacian)
    elif method == 'eigh':
        eigenvalues, eigenvectors = scipy.linalg.eigh(laplacian)
    elif method == 'sparse_eigsh':
        laplacian_sparse = sp.csr_matrix(laplacian)
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(laplacian_sparse, k=k, which='SM')
    elif method == 'sparse_eigs':
        laplacian_sparse = sp.csr_matrix(laplacian)
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(laplacian_sparse, k=k, which='SM')
    else:
        raise ValueError(f"Unknown method: {method}")

    eigenvalues = eigenvalues.astype(np.float64)
    eigenvectors = eigenvectors.astype(np.float64)
    return eigenvalues, eigenvectors


def time_comparison_sparse(N_array, L, shape_func = generate_circle_grid, shape_str = "circle"):
    """
    Compare the time to calculate the eigenvalues using a full matrix and a sparse matrix.

    Parameters:
        N_array (ndarray): An array of different values for N.
        L (float): The diameter of the circle, the length of the side of the square, or the length of the shorter side of the rectangle.
        shape_func (function): The function that generates the grid for the shape.
        shape_str (str): The name of the chosen shape.

    Returns:
        tuple: (sparse_time_array, full_time_array)
            - sparse_time_array (ndarray): The time to calculate the eigenvalues using a sparse matrix.
            - full_time_array (ndarray): The time to calculate the eigenvalues using a full matrix
    """
    num_tests_per_N = 50
    sparse_time_array = []
    full_time_array = []
    for N in N_array:
        average_sparse_time = 0
        average_full_time = 0

        laplacian_shape, in_shape, x, y = create_laplacian_shape(N, L, shape_str, shape_func)
        sprs_laplacian = sp.csr_matrix(laplacian_shape)
        
        for i in range(num_tests_per_N):
            #measure time
            start_time_1 = time.time()
            spla.eigs(sprs_laplacian, k=22)
            sparse_time = time.time() - start_time_1

            start_time_2 = time.time()
            spla.eigs(laplacian_shape, k=22)
            full_matrix_time = time.time() - start_time_2

            average_sparse_time += sparse_time
            average_full_time += full_matrix_time

        average_sparse_time /= num_tests_per_N
        average_full_time /= num_tests_per_N
        sparse_time_array.append(average_sparse_time)
        full_time_array.append(average_full_time)

        print(f"The time to calculate full matrix is {full_matrix_time}, the time to calculate sparse matrix is {sparse_time}")
    return sparse_time_array, full_time_array
