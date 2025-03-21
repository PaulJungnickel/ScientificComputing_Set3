import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse
import scipy.sparse.linalg as spla
import time

from shape_creator import *
from laplacian_creator import *
from drum_eigenmodes import *


def plot_laplacian(laplacian):
    plt.imshow(laplacian, cmap= "magma")
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.colorbar()
    plt.show()

def plot_time_comparison(sparse_time_array, full_time_array, N_array):
    plt.figure()
    plt.plot(N_array, sparse_time_array, color = "blue", label = "Sparse matrix calculation", marker = "o")
    plt.plot(N_array, full_time_array, color = "red", label = "Dense matrix calculation", marker = "o")
    plt.xlabel("Different values for N")
    plt.ylabel("Time in seconds")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

def plot_eigenfrequencies_boxplot(N, L_range, in_shape, shape_str):
    fig, ax = plt.subplots(figsize=(10, 5))

    all_frequencies = []

    for L in L_range:
        laplacian = construct_laplacian(N, in_shape, L/N)
        eigenvalues, _ = eigenvalues_and_eigenvectors(laplacian, 'sparse_eigs')

        eigenfrequencies = np.sqrt(-np.real(eigenvalues))  
        all_frequencies.append(eigenfrequencies)

    all_frequencies = np.array(all_frequencies)

    # Create a box plot
    ax.boxplot(all_frequencies.T, positions=L_range, widths=0.1, patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.5))

    ax.set_xlabel("L", fontsize=16)
    ax.set_ylabel("Eigenfrequency λ", fontsize=16)
    plt.xticks(rotation=90)
    plt.show()

def plot_eigenmodes(eigenvectors, eigenvalues, shape_mask, N):
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    sorted_eigenfrequencies = np.sqrt(-np.real(sorted_eigenvalues))

    for i in range(N):
        mode = np.zeros((N, N))
        mode[shape_mask] = sorted_eigenvectors[:, i]
        plt.imshow(mode)
        plt.colorbar()
        plt.title(f"Eigenfrequency: {sorted_eigenfrequencies[i]}")
        plt.show()

def influence_of_L(N, L_array, shape_func = generate_circle_grid, shape_str = "circle"):
    
    plt.figure(figsize=(6,3))
    for shape_func, shape_str in zip(shape_func, shape_str):
        median_eigenfreq_array = []
        for L in L_array:
            laplacian_shape, in_shape, x, y = create_laplacian_shape(N, L, shape_str, shape_func)
            eigenvalues, _ = eigenvalues_and_eigenvectors(laplacian_shape, 'eig')
            eigenvalues = eigenvalues.astype(np.float64)
            eigenfreq = np.sqrt(-np.real(eigenvalues))
            median_eigenfreq = np.min(eigenfreq)
            median_eigenfreq_array.append(median_eigenfreq)
        #print(eigenfreq)
        
        plt.plot(L_array, median_eigenfreq_array, marker = "o", label = shape_str)

    plt.legend()
    plt.xlabel("L values")
    plt.ylabel("Eigenfrequency")
    plt.grid()
    plt.show()

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
