import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

def circle_shape_laplacian(N, L):
    R = L / 2
    h = L / N

    y, x = np.meshgrid(np.linspace(-R, R, N), np.linspace(-R, R, N))
    in_circle = x**2 + y**2 <= R**2

    indices = np.arange(N * N).reshape(N, N)
    in_circle_indices = indices[in_circle].flatten()
    n_points = len(in_circle_indices)

    laplacian = np.zeros([n_points, n_points])
    for i, idx in enumerate(in_circle_indices):
        row = idx // N
        col = idx % N

        laplacian[i, i] = -4

        for j, k in [(-1,0), (1,0), (0,-1), (0,1)]: # Check neighbors
            row_mod, col_mod = row+j, col+k # Neighbor index

            if 0 <= row_mod < N and 0 <= col_mod < N: # Check if neighbor is in the grid
                neighbor_idx = row_mod * N + col_mod # Neighbor index
                if neighbor_idx in in_circle_indices: # Check if neighbor is in the circle
                    laplacian[i, in_circle_indices == neighbor_idx] = 1

    laplacian /= h**2
    return laplacian, in_circle 

def plot(laplacian):
    plt.imshow(laplacian)
    plt.colorbar()
    plt.show()

def eigenvalues_and_eigenvectors(N,L, method):
    """
    Args: 
        method: scipy.linalg.eigh or scipy.linalg.eig or scipy.sparse.linalg.eigs
    """
    laplacian, _ = circle_shape_laplacian(N,L)
    eigenvalues, eigenvectors = method(laplacian)

    return eigenvalues, eigenvectors

def visualise_circle(eigenvalues, eigenvectors,N,L):

    _, in_circle = circle_shape_laplacian(N,L)
    for i in range(N):
        mode = np.zeros((N, N))  
        mode[in_circle] = eigenvectors[:, i] 
        corresponding_eigenvalue = eigenvalues[i]
        plt.imshow(mode)
        plt.title(f"{corresponding_eigenvalue}")
        #plt.colorbar()
        plt.savefig(f"results/Comparison_eigs_functions/Eigh/{corresponding_eigenvalue}.jpg")
    print("All figures saved")

def influence_of_L():
    L_array = [1,2,3,4,5,6,7,8,9,10,11,12]
    max_eigenfreq_array = []
    min_eigenfreq_array = []
    eigenfreq_boxplot = []

    for L in L_array:
        laplacian, _ = circle_shape_laplacian(25,L)
        eigenvalues, _ = scipy.linalg.eig(laplacian)
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
    plt.savefig("results/eigenfrequncies for different L values")
    plt.show()

def time_comparison_sparse(N_array):
    sparse_time_array = []
    full_time_array = []
    for N in N_array:
        laplacian, _ = circle_shape_laplacian(N, 1)
        sprs_laplacian = sp.csr_matrix(laplacian)
        
        #measure time
        start_time_1 = time.time()
        spla.eigs(sprs_laplacian, k=22)
        sparse_time = time.time() - start_time_1

        start_time_2 = time.time()
        spla.eigs(laplacian, k=22)
        full_matrix_time = time.time() - start_time_2

        sparse_time_array.append(sparse_time)
        full_time_array.append(full_matrix_time)

        print(f"The time to calculate full matrix is {full_matrix_time}, the time to calculate sparse matrix is {sparse_time}")
    return sparse_time_array, full_time_array

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

def main():
    #point A - visualize
    laplacian,_ = circle_shape_laplacian(5,1)
    plot(laplacian)

    #point A - plot eigenvectors
    #eigenvalues_and_eigenvectors(25,1)



    #point B
    N_array = [10,20,30,40,50,60,70,80,90,100]
    sparse_time_array, full_time_array = time_comparison_sparse(N_array)
    plot_time_comparison(sparse_time_array, full_time_array, N_array)


if __name__ == "__main__":
    main()


