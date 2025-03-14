import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

N = 25
L = 1.0 
R = L / 2 
h = L / N  


y, x = np.meshgrid(np.linspace(-R, R, N), np.linspace(-R, R, N))
print("y", y)
print("x", x)
print(x.shape)
print(y.shape)
in_circle = x**2 + y**2 <= R**2
print(in_circle)




indices = np.arange(N * N).reshape(N, N)
print(indices)
in_circle_indices = indices[in_circle].flatten()
print(in_circle_indices)
n_points = len(in_circle_indices)
print(n_points)

laplacian = np.zeros([n_points, n_points])
for i, idx in enumerate(in_circle_indices):
    print(i, " ", idx)
    row = idx // N
    col = idx % N
    print(row, col)
    print(i)

    laplacian[i, i] = -4


    for j, k in [(-1,0), (1,0), (0,-1), (0,1)]: # Check neighbors
        row_mod, col_mod = row+j, col+k # Neighbor index

        if 0 <= row_mod < N and 0 <= col_mod < N: # Check if neighbor is in the grid
            neighbor_idx = row_mod * N + col_mod # Neighbor index
            if neighbor_idx in in_circle_indices: # Check if neighbor is in the circle
                laplacian[i, in_circle_indices == neighbor_idx] = 1
                print("in_circle", in_circle_indices)
                print("neighbor_idx", neighbor_idx)
                print("bool", in_circle_indices == neighbor_idx)
                print("lap", laplacian)

laplacian /= h**2

plt.imshow(laplacian)
plt.colorbar()
plt.show()

eigenvalues, left_eigenvectors, right_eigenvectors = scipy.linalg.eig(laplacian, left=True, right=True)

eigenvalues = eigenvalues.astype(np.float64)
left_eigenvectors = left_eigenvectors.astype(np.float64)
right_eigenvectors = right_eigenvectors.astype(np.float64)


for i in range(N):
    mode = np.zeros((N, N))  
    mode[in_circle] = right_eigenvectors[:, i] 
    plt.imshow(mode)
    plt.colorbar()
    plt.title(f"{i}th Eigenmode")
    plt.show()

for i in range(N):
    mode = np.zeros((N, N))  
    mode[in_circle] = left_eigenvectors[:, i] 
    plt.imshow(mode)
    plt.colorbar()
    plt.title(f"{i}th Eigenmode")
    plt.show()
    



"""mode = np.zeros((N, N))  
print(mode)
print(vr)
print(mode[in_circle])
mode[in_circle] = vr[:, 0] 
print(mode[in_circle])
print(mode)
print(in_circle)
print(vr[:, 0])
plt.imshow(mode)
plt.colorbar()
plt.title("First Eigenmode")
plt.show()




print(w)
print(vl)
print(vr)"""