import numpy as np

def create_laplacian_shape(N, L, shape, generate_grid_func):
    """
    Creates a Laplacian matrix for a given shape.

    Parameters:
        N (int): Number of discretization steps. 
        L (float): For circle - diameter, for square - length of the side, for rectangle - length of the shorter side. 
        shape (str): The name of the chosen shape. It can be: "circle", "square", "rectangle".
        generate_grid_func (function): A function that generates chosen shape grid.

    Returns:
    tuple: (laplacian, in_shape, x, y)
        - laplacian (ndarray): The constructed Laplacian matrix.
        - in_shape (ndarray): A boolean mask indicating the points within the shape.
        - x (ndarray): X-coordinates of the grid.
        - y (ndarray): Y-coordinates of the grid.
    """
    print(shape.upper())
    x, y, in_shape, h = generate_grid_func(N, L)
    laplacian = construct_laplacian(N, in_shape, h)

    return laplacian, in_shape, x, y

def construct_laplacian(N, shape_mask, h):
    """
    Constructs a laplacian matrix for a given shape.

    Parameters:
        N (int): Number of discretization steps.
        shape_mask (ndarray): Boolean mask indicating the points within the shape with True and outside the shape with False.
        h (float): Grid spacing.

    Returns:
        laplacian (ndarray): The constructed Laplacian matrix.
    """
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