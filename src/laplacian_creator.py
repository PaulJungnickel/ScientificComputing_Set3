import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def create_laplacian_shape(N, L, shape, generate_grid_func):
    x, y, in_shape, h = generate_grid_func(N, L)
    laplacian = construct_laplacian(N, in_shape, h)

    return laplacian, in_shape, x, y

def construct_laplacian(N, shape_mask, h):
    indices = np.arange(N * N).reshape(N, N)
    shape_indices = indices[shape_mask].flatten()

    n_points = len(shape_indices)
    laplacian = np.zeros([n_points, n_points])
    for i, idx in enumerate(shape_indices):
        row = idx // N
        col = idx % N
        laplacian[i, i] = -4
        for j, k in [(-1,0), (1,0), (0,-1), (0,1)]:
            row_mod, col_mod = row+j, col+k
            if 0 <= row_mod < N and 0 <= col_mod < N:
                neighbor_idx = row_mod * N + col_mod
                if neighbor_idx in shape_indices:
                    laplacian[i, shape_indices == neighbor_idx] = 1
    laplacian /= h**2
    return laplacian

def create_sparse_laplacian_shape(N, L, shape, generate_grid_func):
    x, y, in_shape, h = generate_grid_func(N, L)
    laplacian = construct_sparse_laplacian(N, in_shape, h)

    return laplacian, in_shape, x, y

def construct_sparse_laplacian(N, shape_mask, h):
    indices = np.arange(N * N).reshape(N, N)
    shape_indices = indices[shape_mask].flatten()

    n_points = len(shape_indices)
    laplacian = lil_matrix((n_points, n_points))
    for i, idx in enumerate(shape_indices):
        row = idx // N
        col = idx % N
        laplacian[i, i] = -4
        for j, k in [(-1,0), (1,0), (0,-1), (0,1)]:
            row_mod, col_mod = row+j, col+k
            if 0 <= row_mod < N and 0 <= col_mod < N:
                neighbor_idx = row_mod * N + col_mod
                if neighbor_idx in shape_indices:
                    laplacian[i, shape_indices == neighbor_idx] = 1
    laplacian = laplacian.tocsr()
    laplacian /= h**2
    return laplacian