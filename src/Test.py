import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

#square matrix 
# matrix size is LxL

def matrix_square(matrix_size):
    matrix = np.zeros((matrix_size*matrix_size, matrix_size*matrix_size))
    for rows in range(matrix_size*matrix_size):
        for columns in range(matrix_size*matrix_size):
            if rows == columns:
                matrix[rows, columns] = -4
    for rows in range(matrix_size*matrix_size - 1):
        if rows%matrix_size ==matrix_size -1:
            matrix[rows, rows + 1] = 0
            matrix[rows, rows - 1] = 1
        elif rows%matrix_size == 0:
            matrix[rows, rows -1 ] = 0
            matrix[rows, rows + 1] = 1
        else:
            matrix[rows, rows + 1] = 1
            matrix[rows, rows - 1] = 1
    matrix[matrix_size*matrix_size - 1, matrix_size*matrix_size - 2] = 1
    for rows in range(matrix_size*matrix_size - matrix_size):
        matrix[rows, rows + matrix_size] = 1
    for rows in range(matrix_size, matrix_size*matrix_size):
        matrix[rows, rows - matrix_size] = 1
    return matrix

# #plotting
# square = matrix_square(4) 
# plt.imshow(square)
# plt.colorbar()
# plt.show()

#rectangle matrix 
# matrix size is Lx2L
def matrix_rectangle(rows, columns):
    size = rows*columns
    matrix = np.zeros((size, size))
    for row in range(size):
        for column in range(size):
            if row == column:
                matrix[row, column] = -4
    for row in range(size-1):
        if row%columns == columns - 1:
            matrix[row, row + 1] = 0
            matrix[row, row - 1] = 1
        elif row%columns == 0:
            matrix[row, row -1 ] = 0
            matrix[row, row + 1] = 1
        else:
            matrix[row, row + 1] = 1
            matrix[row, row - 1] = 1
    matrix[size - 1, size - 2] = 1
    for row in range(size - columns):
        matrix[row, row + columns] = 1
    for row in range(columns, size):
        matrix[row, row - columns] = 1
    return matrix

#plotting
test3 = matrix_rectangle(3,6)
plt.imshow(test3)
plt.colorbar()
plt.show()

eigenvalues, eigenvectors = eig(test3)
plt.imshow(eigenvectors)
plt.show()
print(eigenvectors)
print(eigenvalues)

# circle matrix 
#visualise
def circle_matrix(matrix_size):
    matrix = np.zeros((matrix_size, matrix_size))
    center = (matrix_size - 1) / 2
    radius = matrix_size // 2

    for i in range(matrix_size):
        for j in range(matrix_size):
            # Check if the point is within the circle
            if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                matrix[i, j] = 1

    return matrix

# circle_matrix =circle_matrix()
# plt.imshow(circle_matrix)
# plt.colorbar()
# plt.show()

neighbors = [(1,0), (-1,0), (0,1), (0,-1)]
circle = circle_matrix(5)
size = np.count_nonzero(circle)
print(size)

