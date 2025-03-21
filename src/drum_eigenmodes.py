import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

from shape_creator import *
from laplacian_creator import *

def eigenvalues_and_eigenvectors(laplacian, method, k=22):
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
    sparse_time_array = []
    full_time_array = []
    for N in N_array:
        laplacian_shape, in_shape, x, y = create_laplacian_shape(N, L, shape_str, shape_func)
        sprs_laplacian = sp.csr_matrix(laplacian_shape)
        
        #measure time
        start_time_1 = time.time()
        spla.eigs(sprs_laplacian, k=22)
        sparse_time = time.time() - start_time_1

        start_time_2 = time.time()
        spla.eigs(laplacian_shape, k=22)
        full_matrix_time = time.time() - start_time_2

        sparse_time_array.append(sparse_time)
        full_time_array.append(full_matrix_time)

        print(f"The time to calculate full matrix is {full_matrix_time}, the time to calculate sparse matrix is {sparse_time}")
    return sparse_time_array, full_time_array
