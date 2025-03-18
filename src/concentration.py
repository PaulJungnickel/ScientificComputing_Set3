import numpy as np
import scipy
import scipy.sparse as sp
import matplotlib.pyplot as plt

def steady_state_concentration(x, y, N, shape_mask, laplacian):
    indices = np.arange(x.size).reshape(x.shape) # 2D indices of the full grid
    shape_indices = indices[shape_mask].flatten() # 1D indices of the shape
    n_points = len(shape_indices) # Number of points in the shape

    b = np.zeros(n_points)

    source_x, source_y = 0.6, 1.2
    source_idx = np.argmin(np.abs(x[shape_mask] - source_x)  + np.abs(y[shape_mask] - source_y) )
    print(source_idx)
    
    b[source_idx] = 1

    laplacian[source_idx] = 0
    laplacian[source_idx, source_idx] = 1

    sparse_laplacian = sp.csr_matrix(laplacian)

    c = sp.linalg.spsolve(sparse_laplacian, b)

    #c = scipy.linalg.solve(laplacian, b)
    
    concentration_grid = np.zeros((N, N))
    concentration_grid[shape_mask] = c

    return concentration_grid

def plot_concentration(concentration, in_shape):
    plt.figure(figsize=(10, 10))
    plt.imshow(concentration.T, cmap='magma', origin='lower')
    plt.colorbar()
    plt.contour(in_shape, colors='white', linestyles='dashed')
    plt.show()