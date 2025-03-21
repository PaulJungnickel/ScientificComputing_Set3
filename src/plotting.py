import matplotlib.pyplot as plt
from matplotlib import colors
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
    """
    Plot the Laplacian matrix.

    Parameters:
        laplacian (ndarray): The Laplacian matrix.
    """
    plt.figure(figsize=(6, 4))

    divnorm = colors.TwoSlopeNorm(vmin=-np.max(np.abs(laplacian)), vcenter=0, vmax = np.max(np.abs(laplacian)))

    plt.imshow(laplacian, cmap= "seismic", norm=divnorm)
    plt.xticks()
    plt.yticks()
    plt.colorbar()
    
    plt.tight_layout()
    
    plt.show()

def plot_time_comparison(sparse_time_array, full_time_array, N_array):
    """
    Plot the time comparison between the sparse and full matrix calculation.

    """
    plt.figure(figsize=(6, 3))
    plt.plot(N_array, sparse_time_array, color = "blue", label = "Sparse matrix calculation", marker = "o")
    plt.plot(N_array, full_time_array, color = "red", label = "Dense matrix calculation", marker = "o")
    plt.xlabel("N")
    plt.ylabel("Time (s))")
    plt.yscale("log")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.show()

def plot_eigenfrequencies_boxplot(N, L_range, in_shape, shape_str):
    fig, ax = plt.subplots(figsize=(6, 3))

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

    plt.tight_layout()

    plt.show()

def plot_eigenmodes(eigenvectors, eigenvalues, shape_mask, N):
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    sorted_eigenfrequencies = np.sqrt(-np.real(sorted_eigenvalues))

    num_modes = min(N, sorted_eigenvectors.shape[1])

    for i in range(num_modes):
        mode = np.zeros((N, N))
        mode[shape_mask] = sorted_eigenvectors[:, i]

        plt.figure(figsize=(6, 4))
        plt.imshow(mode)
        plt.colorbar()
        rounded_eigenfrequency = np.round(sorted_eigenfrequencies[i], 7)
        plt.title(f"Eigenfrequency: {rounded_eigenfrequency}")

        plt.tight_layout()

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

def plot_concentration(concentration, in_shape, L):
    """
    Plot the concentration distribution.

    Parameters:
        concentration (ndarray): The concentration distribution.
        in_shape (ndarray): A boolean mask indicating the points within the shape.
        L (float): Size of the grid.
    """
    N = concentration.shape[0]
    extent = [-L/2, L/2, -L/2, L/2] 

    plt.figure(figsize=(4, 4))
    plt.imshow(concentration.T, cmap='hot', origin='lower', extent=extent)
    cbar = plt.colorbar(label="Concentration", fraction=0.046, pad=0.04)
    
    plt.contour(in_shape.T, colors='white', linestyles='dashed', extent=extent)
    
    plt.tight_layout()
    plt.show()
