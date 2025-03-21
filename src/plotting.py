import matplotlib.pyplot as plt
import numpy as np
import scipy

from shape_creator import *
from laplacian_creator import *
from drum_eigenmodes import *


def plot_laplacian(laplacian):
    plt.imshow(laplacian)
    plt.colorbar()
    
    plt.tight_layout()
    
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

    plt.tight_layout()

    plt.show()

def plot_eigenfrequencies_boxplot(N, L_range, in_shape, shape_str):
    fig, ax = plt.subplots(figsize=(8, 4))

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
        plt.imshow(mode)
        plt.colorbar()
        plt.title(f"Eigenfrequency: {sorted_eigenfrequencies[i]}")

        plt.tight_layout()

        plt.show()

def influence_of_L(N, L_array, shape_func = generate_circle_grid, shape_str = "circle"):
    max_eigenfreq_array = []
    min_eigenfreq_array = []
    eigenfreq_boxplot = []

    for L in L_array:
        laplacian_shape, in_shape, x, y = create_laplacian_shape(N, L, shape_str, shape_func)
        eigenvalues, _ = scipy.linalg.eig(laplacian_shape)
        eigenfreq = np.sqrt(-np.real(eigenvalues))
        max_eigenfreq, min_eigenfreq = np.max(eigenfreq), np.min(eigenfreq)
        max_eigenfreq_array.append(max_eigenfreq)
        min_eigenfreq_array.append(min_eigenfreq)
        eigenfreq_boxplot.append(eigenfreq)

    #lineplot
    plt.figure(figsize=(10, 10))
    plt.plot(L_array, min_eigenfreq_array, marker = "o", label = "Minimum eigenvalue frequency", color = "blue")
    plt.plot(L_array, max_eigenfreq_array, marker = "s", label = "Maximum eigenvalue frequency", color = "red")
    plt.legend()
    plt.xlabel("L values")
    plt.ylabel("Eigenfrequency")
    plt.grid()
    #plt.savefig("results/eigenfrequncies for different L values")

    plt.tight_layout()
    
    plt.show()



def plot_concentration(concentration, in_shape, L):
    N = concentration.shape[0]
    extent = [-L/2, L/2, -L/2, L/2] 

    plt.figure(figsize=(5, 5))
    plt.imshow(concentration.T, cmap='hot', origin='lower', extent=extent)
    plt.colorbar(label="Concentration")
    
    plt.contour(in_shape.T, colors='white', linestyles='dashed', extent=extent)
    
    # Label axes with real-world coordinates
    #plt.xlabel("x (radius from center)")
    #plt.ylabel("y (radius from center)")
    #plt.title("Steady-State Concentration Distribution")

    plt.tight_layout()
    plt.show()