import numpy as np
import scipy
import scipy.sparse as sp
import matplotlib.pyplot as plt

from plotting import *

def steady_state_concentration(x, y, N, shape_mask, laplacian):
    indices = np.arange(x.size).reshape(x.shape) # 2D indices of the full grid
    shape_indices = indices[shape_mask].flatten() # 1D indices of the shape
    n_points = len(shape_indices) # Number of points in the shape

    b = np.zeros(n_points)

    source_x, source_y = 0.6, 1.2
    x_diff = np.abs(x[shape_mask] - source_x) # Difference in x between source and all points in the shape
    y_diff = np.abs(y[shape_mask] - source_y) # Difference in y between source and all points in the shape
    total_diff = x_diff + y_diff # Total distance between source and all points in the shape
    source_idx = np.argmin(total_diff) # Index of the point in the shape closest to the source
    
    b[source_idx] = 1

    laplacian[source_idx] = 0 # Remove the row corresponding to the source
    y_diff = np.abs(y[shape_mask] - source_y) # Difference in y between source and all points in the shape
    total_diff = x_diff + y_diff # Total distance between source and all points in the shape
    source_idx = np.argmin(total_diff) # Index of the point in the shape closest to the source
    #print(source_idx)
    
    b[source_idx] = 1

    laplacian[source_idx, :] = 0 # Remove the row corresponding to the source
    laplacian[source_idx, source_idx] = 1 # Set the diagonal element corresponding to the source to 1

    #sparse_laplacian = sp.csr_matrix(laplacian)

    c = sp.linalg.spsolve(laplacian, b)

    
    concentration_grid = np.zeros((N, N))
    concentration_grid[shape_mask] = c

    return concentration_grid

def time_comparison_stead_state_diff(N_array, L, shape_func = generate_circle_grid, shape_str = "circle"):
    num_tests_per_N = 25
    sparse_time_array = []
    full_time_array = []
    for N in N_array:
        average_sparse_time = 0
        average_full_time = 0

        laplacian_shape, in_shape, x, y = create_laplacian_shape(N, L, shape_str, shape_func)
        indices = np.arange(x.size).reshape(x.shape) # 2D indices of the full grid
        shape_indices = indices[in_shape].flatten() # 1D indices of the shape
        n_points = len(shape_indices) # Number of points in the shape

        b = np.zeros(n_points)

        source_x, source_y = 0.6, 1.2

        x_diff = np.abs(x[in_shape] - source_x) 
        y_diff = np.abs(y[in_shape] - source_y) 
        total_diff = x_diff + y_diff
        source_idx = np.argmin(total_diff) 

        b[source_idx] = 1

        laplacian_shape[source_idx] = 0 
        laplacian_shape[source_idx, source_idx] = 1 


        sparse_laplacian = sp.csr_matrix(laplacian_shape)

        for i in range(num_tests_per_N):
            

            start_time_1 = time.time()
            c = sp.linalg.spsolve(sparse_laplacian, b)
            sparse_time = time.time() - start_time_1

            start_time_2 = time.time()
            c = scipy.linalg.solve(laplacian_shape, b)
            full_matrix_time = time.time() - start_time_2
            
            average_sparse_time += sparse_time
            average_full_time += full_matrix_time

        average_sparse_time /= num_tests_per_N
        average_full_time /= num_tests_per_N
        sparse_time_array.append(average_sparse_time)
        full_time_array.append(average_full_time)

        print(f"The time to calculate full matrix is {full_matrix_time}, the time to calculate sparse matrix is {sparse_time}")
    return sparse_time_array, full_time_array
