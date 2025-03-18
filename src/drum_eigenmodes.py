import numpy as np
import scipy
import scipy.sparse as sp


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



