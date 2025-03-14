import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

def create_square_matrix(N, L):
    dx = L / (N + 1)  
    NEIGHBORS = [-N, -1, 1, N]
    A = np.zeros([N**2, N**2])
    
    for i in range(N**2):
        row, col = divmod(i, N)

        A[i, i] = -4
        for di in NEIGHBORS:
            if not 0 <= i + di < N**2:
                continue
            if i % N == 0 and di == -1:
                continue
            elif i % N == N - 1 and di == 1:
                continue
            else:
                A[i, i + di] = 1

    return A / dx**2

def plot_matrix(A):
    plt.imshow(A)
    plt.colorbar()
    plt.show()

def compute_eigenmodes(A):
    w, vl, vr = scipy.linalg.eig(A, left=True, right=True)
    w = w.astype(np.float64)
    vr = vr.astype(np.float64)
    vl = vl.astype(np.float64)
    return w, vl, vr

def plot_eigenmodes(w, vl, N):
    for mode in range(N**2):
        grid = vl[:, mode].reshape([N, N])
        plt.imshow(grid)
        plt.colorbar()
        plt.title(w[mode])
        plt.show()

def test_eigenmode():
    N = 11
    L = 1.0
    
    A = create_square_matrix(N, L)


    plot_matrix(A)

    w, vl, vr = compute_eigenmodes(A)
    print(w)
    plot_eigenmodes(w, vl, N)

if __name__ == '__main__':
    test_eigenmode()  # Change shape to 'square', 'rectangle', or 'circle'
