import numpy as np
import time
import scipy
import scipy.sparse as sp
import matplotlib.pyplot as plt

from numba import njit

def create_laplacian_shape(N, L, shape, generate_grid_func):
    print(shape.upper())
    x, y, in_shape, h = generate_grid_func(N, L)
    laplacian = construct_laplacian(N, in_shape, h)

    return laplacian, in_shape

def generate_circle_grid(N, L):
    R = L / 2
    h = L / N
    y, x = np.meshgrid(np.linspace(-R, R, N), np.linspace(-R, R, N))
    in_circle = x**2 + y**2 <= R**2

    return x, y, in_circle, h 


def generate_square_grid(N, L):
    h = L / N
    y, x = np.meshgrid(np.linspace(0, L, N), np.linspace(0, L, N))
    in_square = np.ones((N, N), dtype=bool)
    in_square[0] = False
    in_square[-1] = False
    in_square[:, 0] = False
    in_square[:, -1] = False

    return x, y, in_square, h


def generate_rectangle_grid(N, L):
    h = L / N  
    y, x = np.meshgrid(np.linspace(0, L, N), np.linspace(0, L, N))

    in_rectangle = np.zeros((N, N), dtype=bool)

    x_min, x_max = L/N, L - (L/N)  # Rectangle spans 25% to 75% in x
    y_min, y_max = L * 0.25, L * 0.75  # Rectangle spans 25% to 75% in y

    in_rectangle[(x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)] = True

    return x, y, in_rectangle, h


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

def plot_laplacian(laplacian):
    plt.imshow(laplacian)
    plt.colorbar()
    plt.show()

def plot_eigenmodes(eigenvectors, eigenvalues, shape_mask, N):
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    for i in range(N):
        mode = np.zeros((N, N))
        mode[shape_mask] = sorted_eigenvectors[:, i]
        plt.imshow(mode)
        plt.colorbar()
        plt.title(f"Eigenvalue: {sorted_eigenvalues[i]}")
        plt.show()



def compute_eigenmodes(laplacian, method='eig', k=10):
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

def benchmark_eig_solution(laplacian, N, sol_method):
    start_time = time.time()
    eigenvalues, eigenvectors = compute_eigenmodes(laplacian, method=sol_method, k=N)
    end_time = time.time()
    print(f"Elapsed time for {sol_method}: {end_time - start_time} seconds")

    return eigenvalues, eigenvectors

def plot_shapes_eigenfrequencies_for_L_range(N, L_range, in_shapes, shape_str):
    fig, ax = plt.subplots(figsize=(8, 4))

    for in_shape, shape in zip(in_shapes, shape_str):
        all_frequencies = []
        
        for L in L_range:
            laplacian = construct_laplacian(N, in_shape, L/N)
            eigenvalues, _ = compute_eigenmodes(laplacian, method='sparse_eigs', k=N)

            eigenfrequencies = np.sqrt(-np.real(eigenvalues)) / L  
            
            all_frequencies.append(eigenfrequencies)

        all_frequencies = np.array(all_frequencies)
        ax.plot(L_range, all_frequencies[:, 0], label=f"First mode ({shape})")
        ax.plot(L_range, all_frequencies[:, -1], label=f"Last mode ({shape})")

    ax.set_xlabel("L", fontsize=16)
    ax.set_ylabel("Eigenfrequency $\lambda$", fontsize=16)
    ax.set_title("Eigenfrequencies vs Size L", fontsize=16)
    ax.legend()
    plt.show()
            
def plot_eigenfrequencies_boxplot(N, L_range, in_shape, shape_str):
    fig, ax = plt.subplots(figsize=(10, 5))

    all_frequencies = []

    for L in L_range:
        laplacian = construct_laplacian(N, in_shape, L/N)
        eigenvalues, _ = compute_eigenmodes(laplacian, method='sparse_eigs', k=10)  # Compute 10 modes

        eigenfrequencies = np.sqrt(-np.real(eigenvalues))  
        all_frequencies.append(eigenfrequencies)

    all_frequencies = np.array(all_frequencies)

    # Create a box plot
    ax.boxplot(all_frequencies.T, positions=L_range, widths=0.1, patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.5))

    ax.set_xlabel("L", fontsize=16)
    ax.set_ylabel("Eigenfrequency λ", fontsize=16)
    ax.set_title("Eigenfrequency Distributions for Different Shapes", fontsize=16)
    plt.show()

def plot_eigenfrequencies_distr_n_steps(N_range, L):
    fig, ax = plt.subplots(figsize=(10, 5))

    all_frequencies = []

    for N in N_range:
        laplacian_square, in_square = create_laplacian_shape(N, L, "square", generate_square_grid)
        eigenvalues, _ = compute_eigenmodes(laplacian_square, method='sparse_eigs', k=10)

        eigenfrequencies = np.sqrt(-np.real(eigenvalues))  

        count, bins = np.histogram(eigenfrequencies, bins=10)

        ax.plot(bins[:-1], count, marker='o', label=f"N={N}")

    ax.set_xlabel("Eigenfrequency λ", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.set_title("Eigenfrequency Distribution for Square at L=1.0", fontsize=16)
    ax.legend()
    plt.show()

def plot_eigenfrequency_distribution(N, L_range, in_shape, shape_str):
    # Plot a Histogram of the eigenfrequencies

    fig, ax = plt.subplots(figsize=(10, 5))

    all_frequencies = []

    for L in L_range:
        laplacian = construct_laplacian(N, in_shape, L/N)
        eigenvalues, _ = compute_eigenmodes(laplacian, method='sparse_eigs', k=10)

        eigenfrequencies = np.sqrt(-np.real(eigenvalues))

        counts, bins = np.histogram(eigenfrequencies, bins=10)

        ax.plot(bins[:-1], counts, marker='o', label=f"L={L}")

    ax.set_xlabel("Eigenfrequency λ", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.set_title("Eigenfrequency Distribution for Square", fontsize=16)
    ax.legend()
    plt.show()
        

def main():
    N = 25
    L = 1.0
    methods = ['eig', 'eigh', 'sparse_eigsh', 'sparse_eigs']

    
    laplacian_circle, in_circle = create_laplacian_shape(N, L, "circle", generate_circle_grid)
    #plot_laplacian(laplacian_circle)
    #eigenvalues_circle, eigenvectors_circle = benchmark_eig_solution(laplacian_circle, N, 'sparse_eigs')
    #plot_eigenmodes(eigenvectors_circle, eigenvalues_circle, in_circle, N)

    laplacian_square, in_square = create_laplacian_shape(N, L, "square", generate_square_grid)
    #plot_laplacian(laplacian_square)
    #eigenvalues_square, eigenvectors_square = benchmark_eig_solution(laplacian_square, N, 'sparse_eigs')
    #plot_eigenmodes(eigenvectors_square, eigenvalues_square, in_square, N)

    laplacian_rect, in_rect = create_laplacian_shape(N, L, "rectangle", generate_rectangle_grid)
    #plot_laplacian(laplacian_rect)
    #eigenvalues_rect, eigenvectors_rect = benchmark_eig_solution(laplacian_rect, N, 'sparse_eigs')
    #plot_eigenmodes(eigenvectors_rect, eigenvalues_rect, in_rect, N)"
    

    L_range = np.linspace(0.1, 12.0, 50)
    in_shapes = [in_circle, in_square, in_rect]
    shape_str = ['circle', 'square', 'rectangle']

    #plot_shapes_eigenfrequencies_for_L_range(N, L_range, in_shapes, shape_str)

    #plot_eigenfrequencies_boxplot(N, L_range, in_square, 'square')

    plot_eigenfrequency_distribution(N, L_range, in_square, 'square')


    N_range = np.linspace(10, 100, 10).astype(int)
    L = 1.0
    plot_eigenfrequencies_distr_n_steps(N_range, L)

if __name__ == "__main__":
    main()