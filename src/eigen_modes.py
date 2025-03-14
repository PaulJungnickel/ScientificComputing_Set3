import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

def create_square_matrix(N, L):
    dx = L / (N + 1)  
    NEIGHBORS = [-N, -1, 1, N]
    A = np.zeros([N**2, N**2])
    
    for i in range(N**2):
        row, col = divmod(i, N)
        print(row, col)
        is_boundary = row == 0 or row == N - 1 or col == 0 or col == N - 1
        


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

def create_rectangle_matrix(N, L, aspect_ratio=2):
    dx = L / (N + 1)  
    NEIGHBORS = [-N, -1, 1, N]
    A = np.zeros([N**2, N**2])
    
    width = N
    height = N // aspect_ratio
    
    for i in range(N**2):
        row, col = divmod(i, N)
        is_boundary = row == 0 or row == height - 1 or col == 0 or col == width - 1
        
        if is_boundary:
            A[i, i] = 1  # Boundary condition
            continue

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

def create_circle_matrix(N, L):
    dx = L / (N + 1)  
    NEIGHBORS = [-N, -1, 1, N]
    A = np.zeros([N**2, N**2])
    
    center = (N // 2, N // 2)
    radius = N // 3 
    
    for i in range(N**2):
        row, col = divmod(i, N)
        is_boundary = (row - center[0])**2 + (col - center[1])**2 >= radius**2
        print(is_boundary)
        
        if is_boundary:
            A[i, i] = 0 
            continue

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

def test_eigenmode(shape='square'):
    N = 5
    L = 1.0
    
    if shape == 'square':
        A = create_square_matrix(N, L)
    elif shape == 'rectangle':
        A = create_rectangle_matrix(N, L)
    elif shape == 'circle':
        A = create_circle_matrix(N, L)
    else:
        raise ValueError("Invalid shape specified")

    plot_matrix(A)

    w, vl, vr = compute_eigenmodes(A)
    print(w)
    plot_eigenmodes(w, vl, N)

if __name__ == '__main__':
    test_eigenmode(shape='square')  # Change shape to 'square', 'rectangle', or 'circle'
